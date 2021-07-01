import torch
import networkx as nx
import itertools
import json
from tqdm import tqdm
# from .semmed import merged_relations
from semmed import relations
from semmed import relations_prune
import numpy as np
from scipy import sparse
import pickle
from scipy.sparse import csr_matrix, coo_matrix
from multiprocessing import Pool
from maths import *

cui2id = None
id2cui = None
relation2id = None
id2relation = None

semmed = None
semmed_all = None
semmed_simple = None


def load_resources(semmed_cui_path):
    global cui2id, id2cui, relation2id, id2relation

    with open(semmed_cui_path, "r", encoding="utf8") as fin:
        id2cui = [c.strip() for c in fin]
        print(id2cui[0:10])
    cui2id = {c: i for i, c in enumerate(id2cui)}

    id2relation = relations_prune
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_semmed(semmed_graph_path):
    global semmed, semmed_simple
    semmed = nx.read_gpickle(semmed_graph_path)
    semmed_simple = nx.Graph()
    for u, v, data in semmed.edges(data=True):
        w = 1.0 # initial weight to 1
        if semmed_simple.has_edge(u, v):
            semmed_simple[u][v]['weight'] += w
        else:
            semmed_simple.add_edge(u, v, weight=w)

def concepts_to_adj_matrices_2hop_all_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in semmed_simple.nodes and aid in semmed_simple.nodes:
                if not semmed_simple.has_edge(qid, aid):
                    extra_nodes |= set(semmed_simple[qid]) & set(semmed_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask

def concepts2adj(node_ids):
    global id2relation
    cids = np.array(node_ids, dtype=np.int32)
    n_rel = len(id2relation)
    n_node = cids.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if semmed.has_edge(s_c, t_c):
                for e_attr in semmed[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cids

def concepts_to_adj_matrices_3hop_all_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in semmed_simple.nodes and aid in semmed_simple.nodes:
                for u in semmed_simple[qid]:
                    for v in semmed_simple[aid]:
                        if semmed_simple.has_edge(u, v):  # ac is a 3-hop neighbour of qc
                            extra_nodes.add(u)
                            extra_nodes.add(v)
                        if u == v:  # ac is a 2-hop neighbour of qc
                            extra_nodes.add(u)
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask

def save_nodes_of_2hop_all_pair(data):
    record_idxs, hf_idxs = data
    nodes = set(record_idxs) | set(hf_idxs)
    extra_nodes = set()

    for record_idx in record_idxs:
        for hf_idx in hf_idxs:
            if record_idx != hf_idx and record_idx in semmed_simple.nodes and hf_idx in semmed_simple.nodes:
                if not semmed_simple.has_edge(record_idx, hf_idx):
                    extra_nodes |= set(semmed_simple[record_idx]) & set(semmed_simple[hf_idx])
    extra_nodes = extra_nodes - nodes
    all_nodes = record_idxs | hf_idxs | extra_nodes
    return all_nodes


def save_nodes_of_3hop_all_pair(data):
    record_idxs, hf_idxs = data
    nodes = set(record_idxs) | set(hf_idxs)
    extra_nodes = set()

    for record_idx in nodes:
        for hf_idx in nodes:
            if record_idx != hf_idx and record_idx in semmed_simple.nodes and hf_idx in semmed_simple.nodes:
                for u in semmed_simple[record_idx]:
                    for v in semmed_simple[hf_idx]:
                        if semmed_simple.has_edge(u, v):  # hf_cui is a 3-hop neighbour of record_cui
                            extra_nodes.add(u)
                            extra_nodes.add(v)
                        if u == v:  # hf_cui is a 2-hop neighbour of record_cui
                            extra_nodes.add(u)
    extra_nodes = extra_nodes - nodes
    all_nodes = record_idxs | hf_idxs | extra_nodes
    return all_nodes

def extract_subgraph_cui(grounded_train_path, grounded_dev_path, grounded_test_path, semmed_graph_path, semmed_cui_path, output_path):
    """
    extracting all cui in the 2hop and 3hop paths of the hfdata as the subgraph cui list
    """
    print("extracting subgraph cui from grounded_path...")

    global cui2id, id2cui, relation2id, id2relation, semmed, semmed_simple
    if any(x is None for x in [cui2id, id2cui, relation2id, id2relation]):
        load_resources(semmed_cui_path)
    if any(x is None for x in [semmed, semmed_simple]):
        load_semmed(semmed_graph_path)

    data = []
    semmed_cui = set()

    with open(grounded_train_path, "r", encoding="utf-8") as fin:
        for line in fin:
            dic = json.loads(line)
            record_idxs = set(cui2id[c] for c in dic["medical_records"]["record_cui_list"])
            hf_idxs = set(cui2id[c] for c in dic["heart_diseases"]["hf_cui"])
            record_idxs = record_idxs - hf_idxs
            if (record_idxs, hf_idxs) not in data:
                data.append((record_idxs, hf_idxs))
    with open(grounded_dev_path, "r", encoding="utf-8") as fin:
        for line in fin:
            dic = json.loads(line)
            record_idxs = set(cui2id[c] for c in dic["medical_records"]["record_cui_list"])
            hf_idxs = set(cui2id[c] for c in dic["heart_diseases"]["hf_cui"])
            record_idxs = record_idxs - hf_idxs
            if (record_idxs, hf_idxs) not in data:
                data.append((record_idxs, hf_idxs))
    with open(grounded_test_path, "r", encoding="utf-8") as fin:
        for line in fin:
            dic = json.loads(line)
            record_idxs = set(cui2id[c] for c in dic["medical_records"]["record_cui_list"])
            hf_idxs = set(cui2id[c] for c in dic["heart_diseases"]["hf_cui"])
            record_idxs = record_idxs - hf_idxs
            if (record_idxs, hf_idxs) not in data:
                data.append((record_idxs, hf_idxs))


    res = []
    for i in tqdm(range(len(data))):
        res.append(save_nodes_of_2hop_all_pair(data[i]))

    for cuis in res:
        semmed_cui |= set(cuis)
    semmed_cui_list = list(semmed_cui)

    with open(output_path, "w", encoding="utf-8") as fout:
        for cui in semmed_cui_list:
            fout.write(str(id2cui[cui]) + "\n")

    print(f'extracted subgraph cui saved to {output_path}')
    print()

def generate_adj_data_from_grounded_concepts(grounded_path, semmed_graph_path, semmed_vocab_path, output_path):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    print(f'generating adj data for {grounded_path}...')

    global cui2id, id2cui, relation2id, id2relation, semmed_simple, semmed
    if any(x is None for x in [cui2id, id2cui, relation2id, id2relation]):
        load_resources(semmed_vocab_path)
    if semmed is None or semmed_simple is None:
        load_semmed(semmed_graph_path)

    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            dic = json.loads(line)
            q_ids = set(cui2id[c] for c in dic["medical_records"]["record_cui_list"])
            a_ids = set(cui2id[c] for c in dic["heart_diseases"]["hf_cui"])
            q_ids = q_ids - a_ids
            qa_data.append((q_ids, a_ids))

  
    res = []
    for i in tqdm(range(len(qa_data))):
        res.append(concepts_to_adj_matrices_2hop_all_pair(qa_data[i]))
    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adj data saved to {output_path}')
    print()


if __name__ == "__main__":

    generate_adj_data_from_grounded_concepts("../data/hfdata/grounded/dev_ground.jsonl", "../data/semmed/database_pruned.graph",
                                             "../data/semmed/sub_cui_vocab.txt", "../data/hfdata/graph/dev_graph_adj.pk")
    generate_adj_data_from_grounded_concepts("../data/hfdata/grounded/train_ground.jsonl",
                                             "../data/semmed/database_pruned.graph",
                                             "../data/semmed/sub_cui_vocab.txt", "../data/hfdata/graph/train_graph_adj.pk")
    generate_adj_data_from_grounded_concepts("../data/hfdata/grounded/test_ground.jsonl",
                                             "../data/semmed/database_pruned.graph",
                                             "../data/semmed/sub_cui_vocab.txt", "../data/hfdata/graph/test_graph_adj.pk")
    