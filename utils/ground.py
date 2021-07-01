import json
from tqdm import tqdm
from semmed import relations_prune
import networkx as nx

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
    #relations_prune_ids = [relation2id[c] for c in relations_prune]

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

def search_nodes_of_2hop_all_pair(data):
    record_idxs, hf_idxs = data
    nodes = set(record_idxs) | set(hf_idxs)
    extra_nodes = set()

    for record_idx in record_idxs:
        for hf_idx in hf_idxs:
            if record_idx != hf_idx and record_idx in semmed_simple.nodes and hf_idx in semmed_simple.nodes:
                if not semmed_simple.has_edge(record_idx, hf_idx):
                    extra_nodes |= set(semmed_simple[record_idx]) & set(semmed_simple[hf_idx])
    extra_nodes = extra_nodes - nodes
    hf_to_rm = set()
    for hf in hf_idxs:
        if not (extra_nodes | set(record_idxs)) & set(semmed_simple[hf]):
            hf_to_rm.add(hf)
    # all_nodes = record_idxs | hf_idxs | extra_nodes
    return hf_to_rm


def ground(semmed_vocab_path, semmed_graph_path, hf_mapped_path, output_path):
    global cui2id, id2cui, relation2id, id2relation, semmed_simple, semmed
    if any(x is None for x in [cui2id, id2cui, relation2id, id2relation]):
        load_resources(semmed_vocab_path)
    if semmed is None or semmed_simple is None:
        load_semmed(semmed_graph_path)
    with open(hf_mapped_path, "r", encoding="utf-8") as fin1, open(output_path, "w", encoding="utf-8") as fout:
        lines = [line for line in fin1]
        for line in tqdm(lines, total=len(lines)):
            j = json.loads(line)
            outj = {}
            record_cui = j["medical_records"]["record_cui"]
            records = []
            for visit in record_cui[:]:
                for cui in visit[:]:
                    if cui not in id2cui:
                        visit.remove(cui)
                if len(visit) != 0:
                    records.append(visit)

            if len(records) != 0:
                j["medical_records"]["record_cui"] = records
                j["medical_records"]["record_cui_list"] = [i for li in records for i in li]
                outj["record_cui"] = [i for li in records for i in li]
                hf_cui_list = j["heart_diseases"]["hf_cui"]
                for hf_cui in hf_cui_list[:]:
                    if hf_cui not in id2cui:
                        hf_cui_list.remove(hf_cui)
                qids = set(cui2id[c] for c in j["medical_records"]["record_cui_list"])
                aids = set(cui2id[c] for c in hf_cui_list)
                qids = qids - aids
                data = (qids, aids)
                hf_to_rm = search_nodes_of_2hop_all_pair(data)
                hf_to_rm = set([id2cui[hf] for hf in hf_to_rm])
                hf_cui_list = list(set(hf_cui_list) - hf_to_rm)
                if len(hf_cui_list) != 0:
                    j["heart_diseases"]["hf_cui"] = hf_cui_list
                    outj["hf_cui"] = hf_cui_list
                    fout.write(json.dumps(j) + "\n")


    print(f'grounded cui saved to {output_path}')
    print()



if __name__ == "__main__":
    ground("../data/semmed/cui_vocab.txt", "../data/semmed/database_all.graph", "../data/hfdata/converted/dev.jsonl", "../data/hfdata/grounded/dev_ground.jsonl")
    ground("../data/semmed/cui_vocab.txt", "../data/semmed/database_all.graph", "../data/hfdata/converted/train.jsonl",
           "../data/hfdata/grounded/train_ground.jsonl")
    ground("../data/semmed/cui_vocab.txt", "../data/semmed/database_all.graph", "../data/hfdata/converted/test.jsonl",
           "../data/hfdata/grounded/test_ground.jsonl")