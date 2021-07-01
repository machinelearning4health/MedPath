import json
import pickle

import dgl
import numpy as np
import torch
from tqdm import tqdm
from transformers import (OpenAIGPTTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer)
from modeling import units, rnn_tools

# from utils.tokenization_utils import *

GPT_SPECIAL_TOKENS = ['_start_', '_delimiter_', '_classify_']


class BatchGenerator(object):
    def __init__(self, device, batch_size, indexes, qids, labels, tensors=[], lists=[]):
        self.device = device
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors = tensors
        self.lists = lists

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes])
            batch_tensors = [self._to_device(x[batch_indexes]) for x in self.tensors]
            batch_lists = [self._to_device([x[i] for i in batch_indexes]) for x in self.lists]
            yield tuple([batch_qids, batch_labels, *batch_tensors, *batch_lists])

    def _to_device(self, obj):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item) for item in obj]
        else:
            return obj.to(self.device)


class MultiGPUBatchGenerator(object):
    def __init__(self, device0, device1, batch_size, indexes, qids, labels, tensors0=[], lists0=[], tensors1=[],
                 lists1=[]):
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device0) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]

            yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)


class AdjDataBatchGenerator(object):
    def __init__(self, device, batch_size, indexes, qids, labels, tensors=[], lists=[], adj_empty=None, adj_data=None):
        self.device = device
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors = tensors
        self.lists = lists
        self.adj_empty = adj_empty
        self.adj_data = adj_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        batch_adj = self.adj_empty  # (batch_size, num_choice, n_rel, n_node, n_node)
        batch_adj[:] = 0
        batch_adj[:, :, -1] = torch.eye(batch_adj.size(-1), dtype=torch.float32, device=self.device)
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes])
            batch_tensors = [self._to_device(x[batch_indexes]) for x in self.tensors]
            batch_lists = [self._to_device([x[i] for i in batch_indexes]) for x in self.lists]

            batch_adj[:, :, :-1] = 0
            for batch_id, global_id in enumerate(batch_indexes):
                for choice_id, (i, j, k) in enumerate(self.adj_data[global_id]):
                    batch_adj[batch_id, choice_id, i, j, k] = 1

            yield tuple([batch_qids, batch_labels, *batch_tensors, *batch_lists, batch_adj[:b - a]])

    def _to_device(self, obj):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item) for item in obj]
        else:
            return obj.to(self.device)


class MultiGPUAdjDataBatchGenerator2(object):
    """
    this version DOES NOT add the identity matrix
    tensors0, lists0  are on device0
    tensors1, lists1, adj, labels  are on device1
    """

    def __init__(self, device0, device1, batch_size, indexes, qids, labels,
                 tensors0=[], lists0=[], tensors1=[], lists1=[], tensors2=[], lists2=[], adj_empty=None, adj_data=None):
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        self.tensors2 = tensors2
        self.lists2 = lists2
        self.adj_empty = adj_empty.to(self.device1)
        self.adj_data = adj_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        batch_adj = self.adj_empty  # (batch_size, num_choice, n_rel, n_node, n_node)
        batch_adj[:] = 0
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device0) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device0) for x in self.tensors1]
            batch_tensors2 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors2]
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists1]
            batch_lists2 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists2]

            batch_adj[:] = 0
            for batch_id, global_id in enumerate(batch_indexes):
                for choice_id, (i, j, k) in enumerate(self.adj_data[global_id]):
                    batch_adj[batch_id, choice_id, i, j, k] = 1

            yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1,
                         *batch_tensors2, *batch_lists2, batch_adj[:b - a]])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)


class MultiGPUNxgDataBatchGenerator(object):
    """
    tensors0, lists0  are on device0
    tensors1, lists1, adj, labels  are on device1
    """

    def __init__(self, device0, device1, batch_size, indexes, qids, labels,
                 tensors0=[], lists0=[], tensors1=[], lists1=[], graph_data=None):
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        self.graph_data = graph_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device0) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            # qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]

            flat_graph_data = sum(self.graph_data, [])
            concept_mapping_dicts = []
            acc_start = 0
            for g in flat_graph_data:
                concept_mapping_dict = {}
                for index, cncpt_id in enumerate(g.ndata['cncpt_ids']):
                    concept_mapping_dict[int(cncpt_id)] = acc_start + index
                acc_start += len(g.nodes())
                concept_mapping_dicts.append(concept_mapping_dict)
            batched_graph = dgl.batch(flat_graph_data)
            batched_graph.ndata['cncpt_ids'] = batched_graph.ndata['cncpt_ids'].to(self.device1)

            yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_tensors1, *batch_lists0, *batch_lists1,
                         batched_graph, concept_mapping_dicts])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)


def load_adj_data(adj_pk_path, max_node_num, num_choice, emb_pk_path=None):
    with open(adj_pk_path, 'rb') as fin:
        adj_concept_pairs = pickle.load(fin)

    n_samples = len(adj_concept_pairs)
    adj_data = []
    adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
    concept_ids = torch.zeros((n_samples, max_node_num), dtype=torch.long)
    node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long)

    if emb_pk_path is not None:
        with open(emb_pk_path, 'rb') as fin:
            all_embs = pickle.load(fin)
        emb_data = torch.zeros((n_samples, max_node_num, all_embs[0].shape[1]), dtype=torch.float)

    adj_lengths_ori = adj_lengths.clone()
    for idx, (adj, concepts, qm, am) in tqdm(enumerate(adj_concept_pairs), total=n_samples,
                                             desc='loading adj matrices'):
        num_concept = min(len(concepts), max_node_num)
        adj_lengths_ori[idx] = len(concepts)
        if emb_pk_path is not None:
            embs = all_embs[idx]
            assert embs.shape[0] >= num_concept
            emb_data[idx, :num_concept] = torch.tensor(embs[:num_concept])
            concepts = np.arange(num_concept)
        else:
            concepts = concepts[:num_concept]
        concept_ids[idx, :num_concept] = torch.tensor(concepts)  # note : concept zero padding is disabled

        adj_lengths[idx] = num_concept
        node_type_ids[idx, :num_concept][torch.tensor(qm, dtype=torch.uint8)[:num_concept]] = 0
        node_type_ids[idx, :num_concept][torch.tensor(am, dtype=torch.uint8)[:num_concept]] = 1
        ij = torch.tensor(adj.row, dtype=torch.int64)
        k = torch.tensor(adj.col, dtype=torch.int64)
        n_node = adj.shape[1]
        half_n_rel = adj.shape[0] // n_node
        i, j = ij // n_node, ij % n_node
        mask = (j < max_node_num) & (k < max_node_num)
        i, j, k = i[mask], j[mask], k[mask]
        i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
        adj_data.append((i, j, k))  # i, j, k are the coordinates of adj's non-zero entries

    print('| ori_adj_len: {:.2f} | adj_len: {:.2f} |'.format(adj_lengths_ori.float().mean().item(),
                                                             adj_lengths.float().mean().item()) +
          ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
          ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
                                                      (node_type_ids == 1).float().sum(1).mean().item()))

    concept_ids, node_type_ids, adj_lengths = [x.view(-1, num_choice, *x.size()[1:]) for x in
                                               (concept_ids, node_type_ids, adj_lengths)]
    if emb_pk_path is not None:
        emb_data = emb_data.view(-1, num_choice, *emb_data.size()[1:])
    adj_data = list(map(list, zip(*(iter(adj_data),) * num_choice)))

    if emb_pk_path is None:
        return concept_ids, node_type_ids, adj_lengths, adj_data, half_n_rel * 2 + 1
    return concept_ids, node_type_ids, adj_lengths, emb_data, adj_data, half_n_rel * 2 + 1


def load_lstm_cui_input_tensors(input_jsonl_path, max_seq_length, max_num_pervisit):
    def _truncate_seq(tokens_a, max_length):
        while len(tokens_a) > max_length:
            tokens_a.pop()

    with open("./data/semmed/sub_cui_vocab.txt", 'r', encoding="utf-8") as fin:
        id2cui = [c.strip() for c in fin]
        cui2id = {c: i for i, c in enumerate(id2cui)}
    qids, labels, input_ids, input_lengths = [], [], [], []
    pad_id = len(id2cui)  # last id of embedding
    pad_seq = []
    for i in range(max_num_pervisit):
        pad_seq.append(pad_id)
    with open(input_jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            input_json = json.loads(line)
            qids.append(input_json['id'])
            labels.append(input_json["heart_diseases"]["hf_label"])
            record_ids = []
            for visit in input_json["medical_records"]["record_cui"]:
                visit_ids = []
                for cui in visit:
                    id = cui2id[cui]
                    visit_ids.append(id)
                _truncate_seq(visit_ids, max_num_pervisit)
                for i in range(0, (max_num_pervisit - len(visit_ids))):
                    visit_ids.append(pad_id)
                record_ids.append(visit_ids)
            _truncate_seq(record_ids, max_seq_length)
            input_lengths.append(len(record_ids))
            for j in range(0, (max_seq_length - len(record_ids))):
                record_ids.append(pad_seq)
            input_ids.append(record_ids)
    for l in labels:
        assert l in [0, 1]
    labels = torch.tensor(labels, dtype=torch.long)
    # labels = labels.unsqueeze(1)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_ids = input_ids.unsqueeze(1)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    input_lengths = input_lengths.unsqueeze(1)
    return qids, labels, input_ids, input_lengths


def load_lstm_icd_input_tensors(input_jsonl_path, max_seq_length, max_num_pervisit):
    def _truncate_seq(tokens_a, max_length):
        while len(tokens_a) > max_length:
            tokens_a.pop()

    with open("./data/hfdata/hf_code2idx_new.pickle", "rb") as fin:
        code2id = pickle.load(fin)
    qids, labels, input_ids, input_lengths = [], [], [], []
    pad_id = len(code2id)  # last id of embedding
    # print("num_icd: ", pad_id)
    pad_seq = []
    for i in range(max_num_pervisit):
        pad_seq.append(pad_id)
    with open(input_jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            input_json = json.loads(line)
            qids.append(input_json['id'])
            labels.append(input_json["heart_diseases"]["hf_label"])
            record_ids = []
            for visit in input_json["medical_records"]["record_icd"]:
                visit_ids = []
                for icd in visit:
                    id = code2id[icd]
                    visit_ids.append(id)
                _truncate_seq(visit_ids, max_num_pervisit)
                for i in range(0, (max_num_pervisit - len(visit_ids))):
                    visit_ids.append(pad_id)
                record_ids.append(visit_ids)
            _truncate_seq(record_ids, max_seq_length)
            input_lengths.append(len(record_ids))
            for j in range(0, (max_seq_length - len(record_ids))):
                record_ids.append(pad_seq)
            input_ids.append(record_ids)
    for l in labels:
        assert l in [0, 1]
    labels = torch.tensor(labels, dtype=torch.long)
    # labels = labels.unsqueeze(1)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_ids = input_ids.unsqueeze(1)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    input_lengths = input_lengths.unsqueeze(1)
    return qids, labels, input_ids, input_lengths


def load_lstm_input_tensors(input_jsonl_path, max_seq_length, max_num_pervisit):
    return load_lstm_icd_input_tensors(input_jsonl_path, max_seq_length, max_num_pervisit)


def load_lstm_multi_input_tensors(input_jsonl_path, max_seq_length, max_num_pervisit):
    qids, labels, *encoder_data = load_lstm_cui_input_tensors(input_jsonl_path, max_seq_length, max_num_pervisit)
    qids2, labels2, *encoder_data2 = load_lstm_icd_input_tensors(input_jsonl_path, max_seq_length, max_num_pervisit)
    return qids, labels, encoder_data, encoder_data2


def load_input_tensors(input_jsonl_path, model_type, model_name, max_seq_length, max_num_pervisit, format=[]):
    if model_type in ('lstm',):
        return load_lstm_input_tensors(input_jsonl_path, max_seq_length, max_num_pervisit)
    elif model_type in ('gpt',):
        return load_gpt_input_tensors(input_jsonl_path, max_seq_length)
    elif model_type in ('bert', 'xlnet', 'roberta'):
        return load_bert_xlnet_roberta_input_tensors(input_jsonl_path, model_type, model_name, max_seq_length,
                                                     format=format)


def load_sand_input(input_jsonl_path, max_seq_len, max_num_pervisit):
    with open("./data/hfdata/hf_code2idx_new.pickle", "rb") as fin:
        code2id = pickle.load(fin)
    n_diagnosis_codes = len(code2id)
    diagnosis_codes = []
    labels = []
    qids = []
    time_step = []
    with open(input_jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            input_json = json.loads(line)
            qids.append(input_json['id'])
            record_icd = input_json["medical_records"]["record_icd"]
            time_dis = input_json["medical_records"]["time_distance"]
            label = input_json["heart_diseases"]["hf_label"]
            for i in range(len(record_icd)):
                for j in range(len(record_icd[i])):
                    record_icd[i][j] = code2id[record_icd[i][j]]
            diagnosis_codes.append(record_icd)
            time_step.append(time_dis)
            labels.append(label)
    t_diagnosis_codes, t_mask, t_mask_final, t_mask_code = rnn_tools.pad_matrix_new(diagnosis_codes, n_diagnosis_codes)
    lengths = torch.from_numpy(np.array([len(seq) for seq in t_diagnosis_codes])).unsqueeze(1)
    t_diagnosis_codes = torch.LongTensor(t_diagnosis_codes).unsqueeze(1)
    t_mask_code = torch.Tensor(t_mask_code).unsqueeze(1)
    labels = torch.tensor(labels, dtype=torch.long)
    return qids, labels, t_diagnosis_codes, t_mask_code, lengths


def load_timeline_input(input_jsonl_path, max_seq_len, max_num_pervisit):
    with open("./data/hfdata/hf_code2idx_new.pickle", "rb") as fin:
        code2id = pickle.load(fin)
    n_diagnosis_codes = len(code2id)
    diagnosis_codes = []
    labels = []
    qids = []
    time_step = []
    with open(input_jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            input_json = json.loads(line)
            qids.append(input_json['id'])
            record_icd = input_json["medical_records"]["record_icd"]
            time_dis = input_json["medical_records"]["time_distance"]
            label = input_json["heart_diseases"]["hf_label"]
            for i in range(len(record_icd)):
                for j in range(len(record_icd[i])):
                    record_icd[i][j] = code2id[record_icd[i][j]]
            diagnosis_codes.append(record_icd)
            time_step.append(time_dis)
            labels.append(label)
    for ind in range(len(diagnosis_codes)):
        if len(diagnosis_codes[ind]) > 50:
            diagnosis_codes[ind] = diagnosis_codes[ind][-50:]
            time_step[ind] = time_step[ind][-50:]
    t_diagnosis_codes, t_mask, t_time, t_mask_final = rnn_tools.pad_matrix_time(diagnosis_codes,
                                                                                time_step,
                                                                                n_diagnosis_codes)
    t_diagnosis_codes = torch.LongTensor(t_diagnosis_codes).unsqueeze(1)
    t_mask = torch.FloatTensor(t_mask).unsqueeze(1)
    t_mask_final = torch.FloatTensor(t_mask_final).unsqueeze(1)
    t_time = torch.FloatTensor(t_time).unsqueeze(1)
    labels = torch.tensor(labels, dtype=torch.long)
    return qids, labels, t_diagnosis_codes, t_mask, t_time, t_mask_final


def load_retain_input(input_jsonl_path, max_seq_len, max_num_pervisit):
    with open("./data/hfdata/hf_code2idx_new.pickle", "rb") as fin:
        code2id = pickle.load(fin)
    n_diagnosis_codes = len(code2id)
    diagnosis_codes = []
    labels = []
    qids = []
    time_step = []
    with open(input_jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            input_json = json.loads(line)
            qids.append(input_json['id'])
            record_icd = input_json["medical_records"]["record_icd"]
            time_dis = input_json["medical_records"]["time_distance"]
            label = input_json["heart_diseases"]["hf_label"]
            for i in range(len(record_icd)):
                for j in range(len(record_icd[i])):
                    record_icd[i][j] = code2id[record_icd[i][j]]
            diagnosis_codes.append(record_icd)
            time_step.append(time_dis)
            labels.append(label)
    for ind in range(len(diagnosis_codes)):
        if len(diagnosis_codes[ind]) > 50:
            diagnosis_codes[ind] = diagnosis_codes[ind][-50:]
            time_step[ind] = time_step[ind][-50:]
    t_diagnosis_codes, t_mask = rnn_tools.pad_matrix(diagnosis_codes,
                                                     labels,
                                                     n_diagnosis_codes,
                                                     max_num_pervisit)
    t_diagnosis_codes = torch.LongTensor(t_diagnosis_codes).permute(1, 0, 2).contiguous().unsqueeze(1)
    t_mask = torch.FloatTensor(t_mask).permute(1, 0).contiguous().unsqueeze(1)
    labels = torch.tensor(labels, dtype=torch.long)
    return qids, labels, t_diagnosis_codes, t_mask


def load_gruself_input(input_jsonl_path, max_seq_len, max_num_pervisit):
    with open("./data/hfdata/hf_code2idx_new.pickle", "rb") as fin:
        code2id = pickle.load(fin)
    n_diagnosis_codes = len(code2id)
    diagnosis_codes = []
    labels = []
    qids = []
    time_step = []
    with open(input_jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            input_json = json.loads(line)
            qids.append(input_json['id'])
            record_icd = input_json["medical_records"]["record_icd"]
            time_dis = input_json["medical_records"]["time_distance"]
            label = input_json["heart_diseases"]["hf_label"]
            for i in range(len(record_icd)):
                for j in range(len(record_icd[i])):
                    record_icd[i][j] = code2id[record_icd[i][j]]
            diagnosis_codes.append(record_icd)
            time_step.append(time_dis)
            labels.append(label)
    for ind in range(len(diagnosis_codes)):
        if len(diagnosis_codes[ind]) > 50:
            diagnosis_codes[ind] = diagnosis_codes[ind][-50:]
            time_step[ind] = time_step[ind][-50:]
    diagnosis_codes, mask, mask_final = rnn_tools.pad_matrix_mine(diagnosis_codes, n_diagnosis_codes, max_num_pervisit)
    diagnosis_codes = torch.LongTensor(diagnosis_codes).permute(1, 0, 2).contiguous().unsqueeze(1)
    mask_mult = torch.FloatTensor(mask).permute(1, 0).contiguous()
    mask_mult = mask_mult.unsqueeze(1)
    labels = torch.tensor(labels, dtype=torch.long)
    return qids, labels, diagnosis_codes, mask_mult


def load_retainex_input(input_jsonl_path, max_seq_len, max_num_pervisit):
    with open("./data/hfdata/hf_code2idx_new.pickle", "rb") as fin:
        code2id = pickle.load(fin)
    n_diagnosis_codes = len(code2id)
    diagnosis_codes = []
    labels = []
    qids = []
    time_step = []
    with open(input_jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            input_json = json.loads(line)
            qids.append(input_json['id'])
            record_icd = input_json["medical_records"]["record_icd"]
            time_dis = input_json["medical_records"]["time_distance"]
            label = input_json["heart_diseases"]["hf_label"]
            for i in range(len(record_icd)):
                for j in range(len(record_icd[i])):
                    record_icd[i][j] = code2id[record_icd[i][j]]
            diagnosis_codes.append(record_icd)
            time_step.append(time_dis)
            labels.append(label)
    for ind in range(len(diagnosis_codes)):
        if len(diagnosis_codes[ind]) > 50:
            diagnosis_codes[ind] = diagnosis_codes[ind][-50:]
            time_step[ind] = time_step[ind][-50:]
    t_diagnosis_codes, t_mask, t_time, t_mask_final = rnn_tools.pad_matrix_retainEx(diagnosis_codes,
                                                                                    labels,
                                                                                    time_step, n_diagnosis_codes,
                                                                                    max_seq_len, max_num_pervisit)
    t_diagnosis_codes = torch.LongTensor(t_diagnosis_codes).unsqueeze(1)
    t_mask = torch.FloatTensor(t_mask).unsqueeze(1)
    t_mask_final = torch.FloatTensor(t_mask_final).unsqueeze(1)
    t_time = torch.FloatTensor(t_time).unsqueeze(1)
    labels = torch.tensor(labels, dtype=torch.long)
    return qids, labels, t_diagnosis_codes, t_mask, t_time, t_mask_final


def load_hita_input(input_jsonl_path, max_seq_len):
    with open("./data/hfdata/hf_code2idx_new.pickle", "rb") as fin:
        code2id = pickle.load(fin)
    n_diagnosis_codes = len(code2id)
    diagnosis_codes = []
    labels = []
    qids = []
    time_step = []
    with open(input_jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            input_json = json.loads(line)
            qids.append(input_json['id'])
            record_icd = input_json["medical_records"]["record_icd"]
            time_dis = input_json["medical_records"]["time_distance"]
            label = input_json["heart_diseases"]["hf_label"]
            for i in range(len(record_icd)):
                for j in range(len(record_icd[i])):
                    record_icd[i][j] = code2id[record_icd[i][j]]
            diagnosis_codes.append(record_icd)
            time_step.append(time_dis)
            labels.append(label)
    # diagnosis_codes = np.array(diagnosis_codes)
    # time_step = np.array(time_step)
    diagnosis_codes, time_step = units.adjust_input(diagnosis_codes, time_step, max_seq_len, n_diagnosis_codes)
    lengths = np.array([max_seq_len + 1 for seq in diagnosis_codes])
    seq_time_step = np.array(list(units.pad_time(time_step, max_seq_len + 1)))
    lengths = torch.from_numpy(lengths)
    diagnosis_codes, mask, mask_final, mask_code = units.pad_matrix_new(diagnosis_codes, n_diagnosis_codes,
                                                                        max_seq_len + 1)
    diagnosis_codes = torch.LongTensor(diagnosis_codes)
    mask_mult = torch.ByteTensor(1 - mask).unsqueeze(2)
    mask_final = torch.Tensor(mask_final).unsqueeze(2)
    mask_code = torch.Tensor(mask_code).unsqueeze(3)
    seq_time_step = torch.Tensor(seq_time_step).unsqueeze(2) / 180
    labels = torch.tensor(labels, dtype=torch.long)
    diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths = \
        diagnosis_codes.unsqueeze(1), seq_time_step.unsqueeze(1), mask_mult.unsqueeze(1), \
        mask_final.unsqueeze(1), mask_code.unsqueeze(1), lengths.unsqueeze(1)
    return qids, labels, diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths


def load_info(statement_path: str):
    n = sum(1 for _ in open(statement_path, "r"))
    num_choice = None
    with open(statement_path, "r", encoding="utf-8") as fin:
        ids = []
        labels = []
        for line in fin:
            input_json = json.loads(line)
            labels.append(ord(input_json.get("answerKey", "A")) - ord("A"))
            ids.append(input_json['id'])
            if num_choice is None:
                num_choice = len(input_json["question"]["choices"])
        labels = torch.tensor(labels, dtype=torch.long)

    return ids, labels, num_choice


def load_statement_dict(statement_path):
    all_dict = {}
    with open(statement_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            instance_dict = json.loads(line)
            qid = instance_dict['id']
            all_dict[qid] = {
                'record_icd': instance_dict['medical_records']['record_icd'],
                'record_cui': instance_dict['medical_records']['record_cui'],
                'label': instance_dict['heart_diseases']['hf_label']
            }
    return all_dict
