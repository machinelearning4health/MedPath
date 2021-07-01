import copy

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from modeling import rnn_tools
from modeling.transformer import Encoder as TrEncoder

high_blood_pressure = ['401', '401.0', '401.1', '401.9', '402.0', '402.00', '402.1', '402.10', '402.9', '402.90']
coronary_artery_disease = ['414.00', '414.01', '414.0']
diabetes = ['250', '250.0', '250.00', '250.01', '250.02', '250.03', '250.1', '250.10', '250.11', '250.12', '250.13',
            '250.2', '250.20', '250.21', '250.22', '250.23', '250.3', '250.30', '250.31', '250.32', '250.33',
            '250.4', '250.40', '250.41', '250.42', '250.43', '250.5', '250.50', '250.51', '250.52', '250.53',
            '250.6', '250.60', '250.61', '250.62', '250.63', '250.7', '250.70', '250.71', '250.72', '250.73',
            '250.8', '250.80', '250.81', '250.82', '250.83', '250.9', '250.90', '250.91', '250.92', '250.93']
congenital_heart_defects = ['V13.65']
valvular_heart_disease = ['424.0']
alcohol_use = ['305.0', '305.00', '305.01', '305.02', '305.03']
obseity = ['278', '278.0', '278.00', '278.01', '278.02', '278.03']
smoking = ['305.1', 'V15.82', 'E869.4']
asthma = ['493', '493.0', '493.00', '493.01', '493.02', '493.1', '493.10', '493.11', '493.12',
          '493.2', '493.20', '493.21', '493.22', '493.8', '493.81', '493.82', '493.9', '493.90',
          '493.91', '493.92']
dusts_chemicals = ['V87.2']
abnormal_kidney_structure = ['794.4']

head_injury = ['959.01', '851.00', '851.01', '851.02', '851.03', '851.04', '851.05', '851.06', '851.09', '851.10',
               '851.11', '851.12', '851.13',
               '851.14', '851.15', '851.16', '851.19', '851.20', '851.21', '851.22', '851.23', '851.24', '851.25',
               '851.26', '851.29', '851.30',
               '851.31', '851.32', '851.33', '851.34', '851.35', '851.36', '851.39', '851.40', '851.41', '851.42',
               '851.43', '851.44', '851.45',
               '851.46', '851.49', '851.50', '851.51', '851.52', '851.53', '851.54', '851.55', '851.56', '851.59',
               '851.60', '851.61', '851.62',
               '851.63', '851.64', '851.65', '851.66', '851.69', '851.70', '851.71', '851.72', '851.73', '851.74',
               '851.75', '851.76', '851.79',
               '851.80', '851.81', '851.82', '851.83', '851.84', '851.85', '851.86', '851.89', '851.90', '851.91',
               '851.92', '851.93', '851.94',
               '851.95', '851.96', '851.99', '852.00', '852.01', '852.02', '852.03', '852.04', '852.05', '852.06',
               '852.09', '852.10', '852.11',
               '852.12', '852.13', '852.14', '852.15', '852.16', '852.19', '852.20', '852.21', '852.22', '852.23',
               '852.24', '852.25', '852.26',
               '852.29', '852.30', '852.31', '852.32', '852.33', '852.34', '852.35', '852.36', '852.39', '852.40',
               '852.41', '852.42', '852.43',
               '852.44', '852.45', '852.46', '852.49', '852.50', '852.51', '852.52', '852.53', '852.54', '852.55',
               '852.56', '852.59', '853.00',
               '853.01', '853.02', '853.03', '853.04', '853.05', '853.06', '853.09', '853.10', '853.11', '853.12',
               '853.13', '853.14', '853.15',
               '853.16', '853.19']
stroke = ['433.01', '433.10', '433.11', '433.21', '433.31', '433.81', '433.91', '434.00', '434.01', '434.11', '434.91',
          '436', '430', '431']
seizures = ['780.31', '780.39', '345.0', '345.00', '345.01', '345.1', '345.10', '345.11', '345.2', '345.3', '345.4',
            '345.40', '345.41',
            '345.5', '345.50', '345.51', '345.6', '345.60', '345.61', '345.7', '345.70', '345.71', '345.8', '345.80',
            '345.81', '345.9',
            '345.90', '345.91']
depression = ['296.1', '296.2', '296.3']
sleep_apnea = ['327.21', '327.22', '327.24', '327.25', '327.20', '327.29', '780.51', '780.53', '780.57']
vitamin_nutritional_deficiencies = ['266.1', '266.2', '268', '268.0', '268.1', '268.2', '268.9']


def adjust_input(batch_diagnosis_codes, batch_time_step, max_len, n_diagnosis_codes):
    batch_time_step = copy.deepcopy(batch_time_step)
    batch_diagnosis_codes = copy.deepcopy(batch_diagnosis_codes)
    for ind in range(len(batch_diagnosis_codes)):
        if len(batch_diagnosis_codes[ind]) > max_len:
            batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-(max_len):]
            batch_time_step[ind] = batch_time_step[ind][-(max_len):]
        batch_time_step[ind].append(0)
        batch_diagnosis_codes[ind].append([n_diagnosis_codes - 1])
    return batch_diagnosis_codes, batch_time_step


def adjust_input_time(batch_diagnosis_codes, batch_time_step, max_len, n_diagnosis_codes, ahead):
    batch_time_step = copy.deepcopy(batch_time_step)
    batch_diagnosis_codes = copy.deepcopy(batch_diagnosis_codes)
    for ind in range(len(batch_diagnosis_codes)):
        if len(batch_diagnosis_codes[ind]) > max_len:
            batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-(max_len):]
            batch_time_step[ind] = batch_time_step[ind][-(max_len):]
        batch_time_step[ind] = batch_time_step[ind][0:ahead]
        batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][0:ahead]
        batch_time_step[ind].append(0)
        batch_diagnosis_codes[ind].append([n_diagnosis_codes - 1])

    return batch_diagnosis_codes, batch_time_step


def adjust_input_gru_time(batch_diagnosis_codes, batch_time_step, max_len, n_diagnosis_codes, ahead):
    batch_time_step = copy.deepcopy(batch_time_step)
    batch_diagnosis_codes = copy.deepcopy(batch_diagnosis_codes)
    for ind in range(len(batch_diagnosis_codes)):
        if len(batch_diagnosis_codes[ind]) > max_len:
            batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-(max_len):]
            batch_time_step[ind] = batch_time_step[ind][-(max_len):]
        batch_time_step[ind] = batch_time_step[ind][0:ahead]
        batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][0:ahead]

    return batch_diagnosis_codes, batch_time_step


class PriorEncoder(nn.Module):
    def __init__(self, batch_size, options):
        super(PriorEncoder, self).__init__()
        code2id = options['code2id']
        if options['disease'] == 'hf':
            self.priors = [high_blood_pressure, coronary_artery_disease, diabetes, congenital_heart_defects,
                           valvular_heart_disease, alcohol_use, obseity, smoking]
        if options['disease'] == 'copd':
            self.priors = [smoking, asthma, dusts_chemicals]
        if options['disease'] == 'kidney':
            self.priors = [high_blood_pressure, diabetes, obseity, smoking, abnormal_kidney_structure]
        if options['disease'] == 'amnesia':
            self.priors = [head_injury, alcohol_use, stroke, seizures]
        if options['disease'] == 'dementias':
            self.priors = [high_blood_pressure, diabetes, alcohol_use, obseity, smoking, depression, sleep_apnea,
                           vitamin_nutritional_deficiencies]
        self.dicts = []
        self.batch_size = batch_size
        for prior in self.priors:
            temp = {}
            ind = 0
            for key in prior:
                if key in code2id.keys():
                    temp[code2id[key]] = ind
                    ind += 1
            if len(temp) > 0:
                self.dicts.append(temp)
        self.relu = nn.ReLU()
        self.embedding_layers = torch.nn.ModuleList([torch.nn.Linear(len(x.keys()), 64) for x in self.dicts])

    def forward(self, seq_dignosis_codes, maxlen, final_queries, options, mask):
        batch_size = seq_dignosis_codes.shape[0]
        if options['use_gpu']:
            final_embeddings = torch.zeros((maxlen, batch_size, 64)).cuda()
        else:
            final_embeddings = torch.zeros((maxlen, batch_size, 64))
        for prior_dict, embedding_layer in zip(self.dicts, self.embedding_layers):
            prior_inputs = torch.zeros((maxlen, batch_size, len(prior_dict.keys())), dtype=torch.float32)
            for b_ind, seq in enumerate(seq_dignosis_codes):
                for o_ind, code_bag in enumerate(seq):
                    for code in code_bag:
                        if code in prior_dict.keys():
                            prior_inputs[o_ind, b_ind, prior_dict[code]] = 1
            if options['use_gpu']:
                prior_inputs = prior_inputs.cuda()
            embeddings = self.relu(embedding_layer(prior_inputs))
            final_embeddings += embeddings
        final_embeddings = torch.sum(final_queries * final_embeddings, 2, keepdim=True) / 8
        final_embeddings = final_embeddings * mask - 255 * (1 - mask)
        return torch.softmax(final_embeddings, 0)

    def show_attention(self, options):
        for prior_dict, embedding_layer in zip(self.dicts, self.embedding_layers):
            prior_inputs = torch.ones((1, len(prior_dict.keys())), dtype=torch.float32)
            prior_inputs_neg = torch.zeros((1, len(prior_dict.keys())), dtype=torch.float32)
            if options['use_gpu']:
                prior_inputs = prior_inputs.cuda()
                prior_inputs_neg = prior_inputs_neg.cuda()
            embeddings = self.relu(embedding_layer(prior_inputs))
            embeddings_neg = self.relu(embedding_layer(prior_inputs_neg))
            prior_weights = self.weight_layer(embeddings)
            prior_weights_neg = self.weight_layer(embeddings_neg)
            print(prior_weights.cpu().detach().numpy())
            print(prior_weights_neg.cpu().detach().numpy())


class TimeEncoder(nn.Module):
    def __init__(self, batch_size):
        super(TimeEncoder, self).__init__()
        self.batch_size = batch_size
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight_layer = torch.nn.Linear(64, 64)

    def forward(self, seq_time_step, final_queries, options, mask):
        if options['use_gpu']:
            seq_time_step = torch.Tensor(seq_time_step).permute(1, 0).unsqueeze(2).cuda() / 180
        else:
            seq_time_step = torch.Tensor(seq_time_step).permute(1, 0).unsqueeze(2) / 180
        selection_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        selection_feature = self.relu(self.weight_layer(selection_feature))
        selection_feature = torch.sum(selection_feature * final_queries, 2, keepdim=True) / 8
        selection_feature = selection_feature * mask - 255 * (1 - mask)
        return torch.softmax(selection_feature, 0)


class FeatureEncoder(nn.Module):
    def __init__(self, n_diagnosis_codes):
        super(FeatureEncoder, self).__init__()
        self.n_diagnosis_codes = n_diagnosis_codes
        self.embedding_layer = torch.nn.Linear(256, 256)
        self.weight_layer = torch.nn.Linear(256, 1)
        self.pre_embed = nn.Linear(n_diagnosis_codes, 256)
        self.relu = nn.ReLU()

    def forward(self, seq_dignosis_codes, batch_labels, options, maxlen):
        batch_size = seq_dignosis_codes.shape[0]
        t_diagnosis_codes, t_labels, t_mask = rnn_tools.pad_matrix_mine(seq_dignosis_codes, batch_labels, options)
        if options['use_gpu']:
            x = torch.Tensor(t_diagnosis_codes).cuda()
        else:
            x = torch.Tensor(t_diagnosis_codes)
        x = self.pre_embed(x)
        x = self.relu(x)
        features = torch.tanh(self.embedding_layer(x))
        weights = self.weight_layer(features)
        return features, t_labels, t_mask, torch.softmax(weights, 0)


class FeatureEncoderGRU(nn.Module):

    def __init__(self, n_diagnosis_codes, visit_size, hidden_size):
        super(FeatureEncoderGRU, self).__init__()

        # code embedding layer
        self.embed = nn.Embedding(n_diagnosis_codes+1, visit_size, padding_idx=-1)

        # relu layer
        self.relu = nn.ReLU()

        # gru layer
        self.gru = nn.GRU(visit_size, 128, num_layers=1, batch_first=False, bidirectional=True)

        self.weight_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(1, 0, 2).contiguous()
        x = self.embed(x) # (n_visits, n_samples, visit_size)
        x = x.sum(2)
        x = self.relu(x)

        output, h_n = self.gru(x)  # output (seq_len, batch, hidden_size)
        weight = self.weight_layer(output)
        return output, torch.softmax(weight, dim=0)


class GRUAttentionModel(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(GRUAttentionModel, self).__init__()
        self.prior_encoder = PriorEncoder(batch_size, options)
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = FeatureEncoderGRU(options)
        self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.relu = nn.ReLU()
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        diagnosis_codes, labels, mask, mask_final = rnn_tools.pad_matrix_mine(seq_dignosis_codes, batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.Tensor(diagnosis_codes).cuda()
            mask_mult = torch.Tensor(mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
        else:
            diagnosis_codes = torch.Tensor(diagnosis_codes)
            mask_mult = torch.Tensor(mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
        features, self_weight = self.feature_encoder(diagnosis_codes, mask_mult)

        final_statues = features * mask_final
        final_statues = final_statues.sum(0, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))
        prior_weight = self.prior_encoder(seq_dignosis_codes, maxlen, quiryes, options, mask_mult)
        time_weight = self.time_encoder(seq_time_step, quiryes, options, mask_mult)

        total_weight = prior_weight * time_weight * self_weight * mask_mult
        total_weight = total_weight / (torch.sum(total_weight, 0, keepdim=True) + 1e-8)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 0)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels


class TransformerAttentionModel(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerAttentionModel, self).__init__()
        self.prior_encoder = PriorEncoder(batch_size, options)
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = TrEncoder(options['n_diagnosis_codes'], 51)
        self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.relu = nn.ReLU()
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(rnn_tools.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final = rnn_tools.pad_matrix_mine(seq_dignosis_codes, batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.Tensor(diagnosis_codes).cuda()
            mask_mult = torch.Tensor(mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
        else:
            diagnosis_codes = torch.Tensor(diagnosis_codes)
            mask_mult = torch.Tensor(mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
        features, self_weight = self.feature_encoder(diagnosis_codes, mask_mult, seq_time_step, lengths)

        final_statues = features * mask_final
        final_statues = final_statues.sum(0, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))
        prior_weight = self.prior_encoder(seq_dignosis_codes, maxlen, quiryes, options, mask_mult)
        time_weight = self.time_encoder(seq_time_step, quiryes, options, mask_mult)

        total_weight = prior_weight * time_weight * self_weight * mask_mult
        total_weight = total_weight / (torch.sum(total_weight, 0, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 0)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight


class TransformerAttentionModelFusion(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerAttentionModelFusion, self).__init__()
        self.prior_encoder = PriorEncoder(batch_size, options)
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = TrEncoder(options['n_diagnosis_codes'], 51, num_layers=options['layer'])
        self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 3)
        self.relu = nn.ReLU()
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(rnn_tools.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final = rnn_tools.pad_matrix_mine(seq_dignosis_codes, batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.Tensor(diagnosis_codes).cuda()
            mask_mult = torch.Tensor(mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
        else:
            diagnosis_codes = torch.Tensor(diagnosis_codes)
            mask_mult = torch.Tensor(mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
        features, self_weight = self.feature_encoder(diagnosis_codes, mask_mult, seq_time_step, lengths)

        final_statues = features * mask_final
        final_statues = final_statues.sum(0, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))
        prior_weight = self.prior_encoder(seq_dignosis_codes, maxlen, quiryes, options, mask_mult)
        time_weight = self.time_encoder(seq_time_step, quiryes, options, mask_mult)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = torch.cat((prior_weight, time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight * mask_mult, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 0, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 0)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight


class TransformerSelf(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerSelf, self).__init__()
        self.feature_encoder = TrEncoderNT(options['n_diagnosis_codes'], 51, num_layers=options['layer'])
        self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.relu = nn.ReLU()
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(rnn_tools.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final = rnn_tools.pad_matrix_mine(seq_dignosis_codes, batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.Tensor(diagnosis_codes).cuda()
            mask_mult = torch.Tensor(mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
        else:
            diagnosis_codes = torch.Tensor(diagnosis_codes)
            mask_mult = torch.Tensor(mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
        features, self_weight = self.feature_encoder(diagnosis_codes, mask_mult, seq_time_step, lengths)

        final_statues = features * mask_final
        final_statues = final_statues.sum(0, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))


        total_weight = self_weight * mask_mult
        total_weight = total_weight / (torch.sum(total_weight, 0, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 0)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight


class TransformerTime(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerTime, self).__init__()
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = TrEncoder(options['n_diagnosis_codes'], 51, num_layers=options['layer'])
        self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 2)
        self.relu = nn.ReLU()
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(rnn_tools.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final = rnn_tools.pad_matrix_mine(seq_dignosis_codes, batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.Tensor(diagnosis_codes).cuda()
            mask_mult = torch.Tensor(mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
        else:
            diagnosis_codes = torch.Tensor(diagnosis_codes)
            mask_mult = torch.Tensor(mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
        features, self_weight = self.feature_encoder(diagnosis_codes, mask_mult, seq_time_step, lengths)

        final_statues = features * mask_final
        final_statues = final_statues.sum(0, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))
        time_weight = self.time_encoder(seq_time_step, quiryes, options, mask_mult)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight * mask_mult, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 0, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 0)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight


class TransformerTimePure(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerTimePure, self).__init__()
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = TrEncoder(options['n_diagnosis_codes'], 51, num_layers=options['layer'])
        self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 2)
        self.relu = nn.ReLU()
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(rnn_tools.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final = rnn_tools.pad_matrix_mine(seq_dignosis_codes, batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.Tensor(diagnosis_codes).cuda()
            mask_mult = torch.Tensor(mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
        else:
            diagnosis_codes = torch.Tensor(diagnosis_codes)
            mask_mult = torch.Tensor(mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
        features, self_weight = self.feature_encoder(diagnosis_codes, mask_mult, seq_time_step, lengths)

        final_statues = features * mask_final
        final_statues = final_statues.sum(0, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))

        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = self_weight * mask_mult
        total_weight = total_weight / (torch.sum(total_weight, 0, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 0)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight


class TransformerPrior(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerPrior, self).__init__()
        self.prior_encoder = PriorEncoder(batch_size, options)
        self.feature_encoder = TrEncoderNT(options['n_diagnosis_codes'], 51, num_layers=options['layer'])
        self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 2)
        self.relu = nn.ReLU()
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(rnn_tools.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final = rnn_tools.pad_matrix_mine(seq_dignosis_codes, batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.Tensor(diagnosis_codes).cuda()
            mask_mult = torch.Tensor(mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
        else:
            diagnosis_codes = torch.Tensor(diagnosis_codes)
            mask_mult = torch.Tensor(mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
        features, self_weight = self.feature_encoder(diagnosis_codes, mask_mult, seq_time_step, lengths)

        final_statues = features * mask_final
        final_statues = final_statues.sum(0, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))
        prior_weight = self.prior_encoder(seq_dignosis_codes, maxlen, quiryes, options, mask_mult)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = torch.cat((prior_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight * mask_mult, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 0, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 0)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight


class GRUSelfAttentionModel(nn.Module):
    def __init__(self, n_diagnosis_codes, visit_size, hidden_size, dropout_rate):
        super(GRUSelfAttentionModel, self).__init__()
        self.feature_encoder = FeatureEncoderGRU(n_diagnosis_codes, visit_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, diagnosis_codes, mask_mult):

        mask_mult = mask_mult.permute(1, 0).contiguous().unsqueeze(2)
        features, self_weight = self.feature_encoder(diagnosis_codes)

        total_weight = self_weight * mask_mult
        total_weight = total_weight / (torch.sum(total_weight, 0, keepdim=True) + 1e-8)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 0)
        averaged_features = self.dropout(averaged_features)

        return averaged_features


class BasicPureModel(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(BasicPureModel, self).__init__()
        self.feature_encoder = FeatureEncoder(n_diagnosis_codes)
        self.classify_layer = torch.nn.Linear(256, 2)
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seq_dignosis_codes, batch_labels, options, maxlen):
        features, labels, masks, self_weight = self.feature_encoder(seq_dignosis_codes, batch_labels, options, maxlen)
        if options['use_gpu']:
            mask_mult = torch.Tensor(masks).unsqueeze(2).cuda()
        else:
            mask_mult = torch.Tensor(masks).unsqueeze(2)
        weighted_features = features * mask_mult / torch.sum(mask_mult, 0, keepdim=True)
        averaged_features = torch.sum(weighted_features, 0)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = nn.functional.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()


        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class Timeline(nn.Module):

    def __init__(self, batch_size, embedding_dim, hidden_dim, attention_dim, vocab_size, labelset_size, dropoutrate):
        super(Timeline, self).__init__()
        self.hidden_dim = hidden_dim
        self.batchsi = batch_size
        self.word_embeddings = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * 2, labelset_size)
        self.attention = nn.Linear(embedding_dim, attention_dim)
        self.vector1 = nn.Parameter(torch.randn(attention_dim, 1))
        self.decay = nn.Parameter(torch.FloatTensor([-0.1] * (vocab_size + 1)))
        self.initial = nn.Parameter(torch.FloatTensor([1.0] * (vocab_size + 1)))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.attention_dimensionality = attention_dim
        self.WQ1 = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.WK1 = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.embed_drop = nn.Dropout(p=dropoutrate)

    def forward(self, *sentence, Mode=1):
        numcode = sentence[0].size()[2]
        numvisit = sentence[0].size()[1]
        numbatch = sentence[0].size()[0]
        thisembeddings = self.word_embeddings(sentence[0].view(-1, numcode))
        thisembeddings = self.embed_drop(thisembeddings)
        myQ1 = self.WQ1(thisembeddings)
        myK1 = self.WK1(thisembeddings)
        dproduct1 = torch.bmm(myQ1, torch.transpose(myK1, 1, 2)).view(numbatch, numvisit, numcode, numcode)
        dproduct1 = dproduct1 - sentence[1].view(numbatch, numvisit, 1, numcode) - sentence[1].view(numbatch, numvisit,
                                                                                                    numcode, 1)
        sproduct1 = self.softmax(dproduct1.view(-1, numcode) / np.sqrt(self.attention_dimensionality)).view(-1, numcode,
                                                                                                            numcode)
        fembedding11 = torch.bmm(sproduct1, thisembeddings)
        fembedding11 = (((sentence[1] - (1e+20)) / (-1e+20)).view(-1, numcode, 1) * fembedding11)
        mydecay = self.decay[sentence[0].view(-1)].view(numvisit * numbatch, numcode, 1)
        myini = self.initial[sentence[0].view(-1)].view(numvisit * numbatch, numcode, 1)
        temp1 = torch.bmm(mydecay, sentence[2].view(-1, 1, 1))
        temp2 = self.sigmoid(temp1 + myini)
        vv = torch.bmm(temp2.view(-1, 1, numcode), fembedding11)
        vv = vv.view(numbatch, numvisit, -1).transpose(0, 1)
        lstm_out, hidden = self.lstm(vv)
        mask_final = sentence[3]
        lstm_out_final = lstm_out * mask_final.transpose(0, 1).view(numvisit, numbatch, 1)
        lstm_out_final = lstm_out_final.sum(dim=0)
        label_space = self.hidden2label(lstm_out_final)
        return lstm_out_final


class RetainNN(nn.Module):
    def __init__(self, n_diagnosis_codes, hidden_size, dropout_rate, batch_size, n_labels=2):
        super(RetainNN, self).__init__()
        """
        num_embeddings(int): size of the dictionary of embeddings
        embedding_dim(int) the size of each embedding vector
        """
        self.emb_layer = nn.Embedding(num_embeddings=n_diagnosis_codes+1, embedding_dim=hidden_size, padding_idx=-1)
        self.linear_layer = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.variable_level_rnn = nn.GRU(hidden_size, hidden_size)
        self.visit_level_rnn = nn.GRU(hidden_size, hidden_size)
        self.variable_level_attention = nn.Linear(hidden_size, hidden_size)
        self.visit_level_attention = nn.Linear(hidden_size, 1)
        self.output_dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size, n_labels)

        self.var_hidden_size = hidden_size

        self.visit_hidden_size = hidden_size

        self.n_samples = batch_size
        self.reverse_rnn_feeding = False

    def forward(self, input, mask):
        """
        :param input:
        :param var_rnn_hidden:
        :param visit_rnn_hidden:
        :return:
        """

        v = self.emb_layer(input.permute(1, 0, 2).contiguous())
        v = v.sum(2)
        v = self.linear_layer(v)

        v = self.dropout(v)


        if self.reverse_rnn_feeding:
            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(torch.flip(v, [0]))
            alpha = self.visit_level_attention(torch.flip(visit_rnn_output, [0]))
        else:
            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(v)
            alpha = self.visit_level_attention(visit_rnn_output)
        visit_attn_w = torch.nn.functional.softmax(alpha, dim=0)

        if self.reverse_rnn_feeding:
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(torch.flip(v, [0]))
            beta = self.variable_level_attention(torch.flip(var_rnn_output, [0]))
        else:
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(v)
            beta = self.variable_level_attention(var_rnn_output)
        var_attn_w = torch.tanh(beta)

        attn_w = visit_attn_w * var_attn_w

        c = torch.sum(attn_w * v, dim=0)

        c = self.output_dropout(c)
        output = self.output_layer(c)


        return c
