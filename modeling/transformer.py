import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.init as init
import copy

from modeling import rnn_tools

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
              '493.2', '493.20', '493.21','493.22', '493.8', '493.81', '493.82', '493.9', '493.90',
              '493.91', '493.92']
dusts_chemicals = ['V87.2']
abnormal_kidney_structure = ['794.4']

head_injury = ['959.01', '851.00', '851.01', '851.02', '851.03', '851.04', '851.05', '851.06', '851.09', '851.10', '851.11', '851.12', '851.13',
                   '851.14', '851.15', '851.16', '851.19', '851.20', '851.21', '851.22', '851.23', '851.24', '851.25', '851.26', '851.29', '851.30',
                   '851.31', '851.32', '851.33', '851.34', '851.35', '851.36', '851.39', '851.40', '851.41', '851.42', '851.43', '851.44', '851.45',
                   '851.46', '851.49', '851.50', '851.51', '851.52', '851.53', '851.54', '851.55', '851.56', '851.59', '851.60', '851.61', '851.62',
                   '851.63', '851.64', '851.65', '851.66', '851.69', '851.70', '851.71', '851.72', '851.73', '851.74', '851.75', '851.76', '851.79',
                   '851.80', '851.81', '851.82', '851.83', '851.84', '851.85', '851.86', '851.89', '851.90', '851.91', '851.92', '851.93', '851.94',
                   '851.95', '851.96', '851.99', '852.00', '852.01', '852.02', '852.03', '852.04', '852.05', '852.06', '852.09', '852.10', '852.11',
                   '852.12', '852.13', '852.14', '852.15', '852.16', '852.19', '852.20', '852.21', '852.22', '852.23', '852.24', '852.25', '852.26',
                   '852.29', '852.30', '852.31', '852.32', '852.33', '852.34', '852.35', '852.36', '852.39', '852.40', '852.41', '852.42', '852.43',
                   '852.44', '852.45', '852.46', '852.49', '852.50', '852.51', '852.52', '852.53', '852.54', '852.55', '852.56', '852.59', '853.00',
                   '853.01', '853.02', '853.03', '853.04', '853.05', '853.06', '853.09', '853.10', '853.11', '853.12', '853.13', '853.14', '853.15',
                   '853.16', '853.19']
stroke = ['433.01', '433.10', '433.11', '433.21', '433.31', '433.81', '433.91', '434.00', '434.01', '434.11', '434.91', '436', '430', '431']
seizures = ['780.31', '780.39', '345.0', '345.00', '345.01', '345.1', '345.10', '345.11', '345.2', '345.3', '345.4', '345.40', '345.41',
                '345.5', '345.50', '345.51', '345.6', '345.60', '345.61', '345.7', '345.70', '345.71', '345.8', '345.80', '345.81', '345.9',
                '345.90', '345.91']
depression = ['296.1', '296.2', '296.3']
sleep_apnea = ['327.21', '327.22', '327.24', '327.25', '327.20', '327.29', '780.51', '780.53', '780.57']
vitamin_nutritional_deficiencies = ['266.1', '266.2', '268', '268.0', '268.1', '268.2', '268.9']


class Embedding(torch.nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx,
                                        max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                        sparse=sparse, _weight=_weight)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):

        super(PositionalEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding.astype(np.float32))

        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))


        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):



        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        pos = np.zeros([len(input_len), max_len])
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                pos[ind, pos_ind - 1] = pos_ind
        input_pos = tensor(pos)
        return self.position_encoding(input_pos), input_pos


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


def padding_mask_sand(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=1,
                 model_dim=256,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        self.pre_embedding = nn.Linear(vocab_size, model_dim)
        self.weight_layer = torch.nn.Linear(model_dim, 1)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.time_layer = torch.nn.Linear(64, 256)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, diagnosis_codes, mask, seq_time_step, input_len):
        diagnosis_codes = diagnosis_codes.permute(1, 0, 2)
        seq_time_step = torch.Tensor(seq_time_step).cuda().unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
        mask = mask.permute(1, 0, 2)
        output = self.pre_embedding(diagnosis_codes)
        output += time_feature
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        output += output_pos
        self_attention_mask = padding_mask(ind_pos, ind_pos)

        attentions = []
        outputs = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
            outputs.append(output)
        weight = torch.softmax(self.weight_layer(outputs[-1]), dim=1)
        weight = weight * mask - 255 * (1 - mask)
        output = outputs[-1].permute(1, 0, 2)
        weight = weight.permute(1, 0, 2)
        return output, weight


class EncoderNew(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=1,
                 model_dim=256,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0):
        super(EncoderNew, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        self.pre_embedding = Embedding(vocab_size, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.time_layer = torch.nn.Linear(64, 256)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, diagnosis_codes, mask, mask_code, seq_time_step, input_len):
        seq_time_step = torch.Tensor(seq_time_step).cuda().unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
        output = (self.pre_embedding(diagnosis_codes) * mask_code).sum(dim=2) + self.bias_embedding
        output += time_feature
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        output += output_pos
        self_attention_mask = padding_mask(ind_pos, ind_pos)

        attentions = []
        outputs = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
            outputs.append(output)
        return output

class EncoderEval(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=1,
                 model_dim=256,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0):
        super(EncoderEval, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        self.pre_embedding = Embedding(vocab_size, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        # self.weight_layer = torch.nn.Linear(model_dim, 1)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.time_layer = torch.nn.Linear(64, 256)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, diagnosis_codes, mask, mask_code, seq_time_step, input_len):
        seq_time_step = torch.Tensor(seq_time_step).cuda().unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
        output = (self.pre_embedding(diagnosis_codes) * mask_code).sum(dim=2) + self.bias_embedding
        output += time_feature
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        output += output_pos
        self_attention_mask = padding_mask(ind_pos, ind_pos)

        attentions = []
        outputs = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
            outputs.append(output)
        return output, attention

class EncoderPure(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=1,
                 model_dim=256,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0):
        super(EncoderPure, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        self.pre_embedding = Embedding(vocab_size, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, diagnosis_codes, mask, mask_code, seq_time_step, input_len):
        output = (self.pre_embedding(diagnosis_codes) * mask_code).sum(dim=2) + self.bias_embedding
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        output += output_pos
        self_attention_mask = padding_mask(ind_pos, ind_pos)

        attentions = []
        outputs = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
            outputs.append(output)
        return output


class SAND(nn.Module):
    def __init__(self,
                 vocab_size,
                 model_dim=256,
                 num_heads=4,
                 out_dim=2,
                 dropout=0.0,
                 M=4):
        super(SAND, self).__init__()

        self.encoder_layers = MultiHeadAttention(model_dim, num_heads, dropout)
        self.pre_embedding = Embedding(vocab_size+1, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)
        self.pos_embedding = PositionalEncoding(model_dim, 51)
        self.drop_out = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.M = M

    def padding_mask(self, seq_k, seq_q):
        len_q = seq_q.size(1)
        pad_mask = seq_k.eq(0)
        pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
        return pad_mask

    def forward(self, diagnosis_codes, mask_code, lengths):
        mask_code = mask_code.unsqueeze(3)
        output = (self.pre_embedding(diagnosis_codes) * mask_code).sum(dim=2) + self.bias_embedding
        output_pos, ind_pos = self.pos_embedding(lengths.unsqueeze(1))
        output += output_pos
        self_attention_mask = padding_mask(ind_pos, ind_pos)
        output, attention = self.encoder_layers(output, output, output, self_attention_mask)
        U = torch.zeros((output.size(0), self.M, output.size(2))).cuda()
        lengths = lengths.float()
        for t in range(1, diagnosis_codes.size(1)+1):
            s = self.M * t / lengths
            for m in range(1, self.M+1):
                w = torch.pow(1 - torch.abs(s - m) / self.M, 2)
                U[:, m-1] += w.unsqueeze(-1) * output[:, t-1]
        U = U.sum(1)
        U = self.drop_out(U)
        return U



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

class TimeEncoderNRelu(nn.Module):
    def __init__(self, batch_size):
        super(TimeEncoderNRelu, self).__init__()
        self.batch_size = batch_size
        self.selection_layer = torch.nn.Linear(1, 64)
        self.tanh = nn.Tanh()
        self.weight_layer = torch.nn.Linear(64, 64)

    def forward(self, seq_time_step, final_queries, options, mask):
        if options['use_gpu']:
            seq_time_step = torch.Tensor(seq_time_step).unsqueeze(2).cuda() / 180
        else:
            seq_time_step = torch.Tensor(seq_time_step).unsqueeze(2) / 180
        selection_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        selection_feature = self.weight_layer(selection_feature)
        selection_feature = torch.sum(selection_feature * final_queries, 2, keepdim=True) / 8
        selection_feature = selection_feature.masked_fill_(mask, -np.inf)
        return torch.softmax(selection_feature, 1)

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
            seq_time_step = torch.Tensor(seq_time_step).unsqueeze(2).cuda() / 180
        else:
            seq_time_step = torch.Tensor(seq_time_step).unsqueeze(2) / 180
        selection_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        selection_feature = self.relu(self.weight_layer(selection_feature))
        selection_feature = torch.sum(selection_feature * final_queries, 2, keepdim=True) / 8
        selection_feature = selection_feature.masked_fill_(mask, -np.inf)
        return torch.softmax(selection_feature, 1)

class PriorEncoder(nn.Module):
    def __init__(self, batch_size, options):
        super(PriorEncoder, self).__init__()
        code2id = options['code2id']
        if options['disease'] == 'hf':
            self.priors = [high_blood_pressure, coronary_artery_disease, diabetes, congenital_heart_defects, valvular_heart_disease, alcohol_use, obseity, smoking]
        if options['disease'] == 'copd':
            self.priors = [smoking, asthma, dusts_chemicals]
        if options['disease'] == 'kidney':
            self.priors = [high_blood_pressure, diabetes, obseity, smoking, abnormal_kidney_structure]
        if options['disease'] == 'amnesia':
            self.priors = [head_injury, alcohol_use, stroke, seizures]
        if options['disease'] == 'dementias':
            self.priors = [high_blood_pressure, diabetes, alcohol_use, obseity, smoking, depression, sleep_apnea, vitamin_nutritional_deficiencies]
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
            final_embeddings = torch.zeros((batch_size, maxlen, 64)).cuda()
        else:
            final_embeddings = torch.zeros((batch_size, maxlen, 64))
        for prior_dict, embedding_layer in zip(self.dicts, self.embedding_layers):
            prior_inputs = torch.zeros((batch_size, maxlen, len(prior_dict.keys())), dtype=torch.float32)
            for b_ind, seq in enumerate(seq_dignosis_codes):
                for o_ind, code_bag in enumerate(seq):
                    for code in code_bag:
                        if code in prior_dict.keys():
                            prior_inputs[b_ind, o_ind, prior_dict[code]] = 1
            if options['use_gpu']:
                prior_inputs = prior_inputs.cuda()
            embeddings = self.relu(embedding_layer(prior_inputs))
            final_embeddings += embeddings
        final_embeddings = torch.sum(final_queries * final_embeddings, 2, keepdim=True)/8
        final_embeddings = final_embeddings * mask - 255*(1-mask)
        return torch.softmax(final_embeddings, 1)

class TransformerTime(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerTime, self).__init__()
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = EncoderNew(options['n_diagnosis_codes'] + 1, 51, num_layers=options['layer'])
        self.self_layer = torch.nn.Linear(256, 1)
        self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 2)
        self.relu = nn.ReLU(inplace=True)
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill(mask, -np.inf), dim=1)
        return attention

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(rnn_tools.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final, mask_code = rnn_tools.pad_matrix_new(seq_dignosis_codes,
                                                                                        batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.LongTensor(diagnosis_codes).cuda()
            mask_mult = torch.BoolTensor(1-mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
            mask_code = torch.Tensor(mask_code).unsqueeze(3).cuda()
        else:
            diagnosis_codes = torch.LongTensor(diagnosis_codes)
            mask_mult = torch.BoolTensor(1-mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
            mask_code = torch.Tensor(mask_code).unsqueeze(3)
        features = self.feature_encoder(diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))

        self_weight = self.get_self_attention(features, quiryes, mask_mult)
        time_weight = self.time_encoder(seq_time_step, quiryes, options, mask_mult)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight

class TransformerTimeMix(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerTimeMix, self).__init__()
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = EncoderNew(options['n_diagnosis_codes'] + 1, 51, num_layers=options['layer'])
        self.self_layer = torch.nn.Linear(256, 1)
        self.classify_layer = torch.nn.Linear(256*2, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill(mask, -np.inf), dim=1)
        return attention

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(rnn_tools.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final, mask_code = rnn_tools.pad_matrix_new(seq_dignosis_codes,
                                                                                        batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.LongTensor(diagnosis_codes).cuda()
            mask_mult = torch.BoolTensor(1-mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
            mask_code = torch.Tensor(mask_code).unsqueeze(3).cuda()
        else:
            diagnosis_codes = torch.LongTensor(diagnosis_codes)
            mask_mult = torch.BoolTensor(1-mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
            mask_code = torch.Tensor(mask_code).unsqueeze(3)
        features = self.feature_encoder(diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)

        quiryes = self.relu(self.quiry_layer(final_statues))

        self_weight = self.get_self_attention(features, quiryes, mask_mult)
        time_weight = self.time_encoder(seq_time_step, quiryes, options, mask_mult)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        mix_features = torch.cat([averaged_features, final_statues.squeeze()], 1)
        averaged_features = self.dropout(mix_features)
        predictions = self.classify_layer(mix_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight

class TransformerTimeNRelu(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerTimeNRelu, self).__init__()
        self.time_encoder = TimeEncoderNRelu(batch_size)
        self.feature_encoder = EncoderNew(options['n_diagnosis_codes'] + 1, 51, num_layers=options['layer'])
        self.self_layer = torch.nn.Linear(256, 1)
        self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 2)
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill_(mask, -np.inf), dim=1)
        return attention

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(rnn_tools.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final, mask_code = rnn_tools.pad_matrix_new(seq_dignosis_codes,
                                                                                        batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.LongTensor(diagnosis_codes).cuda()
            mask_mult = torch.BoolTensor(1-mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
            mask_code = torch.Tensor(mask_code).unsqueeze(3).cuda()
        else:
            diagnosis_codes = torch.LongTensor(diagnosis_codes)
            mask_mult = torch.BoolTensor(1-mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
            mask_code = torch.Tensor(mask_code).unsqueeze(3)
        features = self.feature_encoder(diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.quiry_layer(final_statues)

        self_weight = self.get_self_attention(features, quiryes, mask_mult)
        time_weight = self.time_encoder(seq_time_step, quiryes, options, mask_mult)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight

class TransformerTimePrior(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerTimePrior, self).__init__()
        self.prior_encoder = PriorEncoder(batch_size, options)
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = EncoderNew(options['n_diagnosis_codes'] + 1, 51, num_layers=options['layer'])
        self.self_layer = torch.nn.Linear(256, 1)
        self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 3)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill_(mask, -np.inf), dim=1)
        return attention

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(rnn_tools.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final, mask_code = rnn_tools.pad_matrix_new(seq_dignosis_codes,
                                                                                        batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.LongTensor(diagnosis_codes).cuda()
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
            mask_code = torch.Tensor(mask_code).unsqueeze(3).cuda()
        else:
            diagnosis_codes = torch.LongTensor(diagnosis_codes)
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
            mask_code = torch.Tensor(mask_code).unsqueeze(3)
        features = self.feature_encoder(diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))

        prior_weight = self.prior_encoder(seq_dignosis_codes, maxlen, quiryes, options, mask_mult)
        self_weight = self.get_self_attention(features, quiryes, mask_mult)
        time_weight = self.time_encoder(seq_time_step, quiryes, options, mask_mult)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, prior_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight

class TransformerTimeAtt(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerTimeAtt, self).__init__()
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = EncoderPure(options['n_diagnosis_codes'] + 1, 51, num_layers=options['layer'])
        self.self_layer = torch.nn.Linear(256, 1)
        self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill_(mask, -np.inf), dim=1)
        return attention

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(rnn_tools.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final, mask_code = rnn_tools.pad_matrix_new(seq_dignosis_codes,
                                                                                        batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.LongTensor(diagnosis_codes).cuda()
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
            mask_code = torch.Tensor(mask_code).unsqueeze(3).cuda()
        else:
            diagnosis_codes = torch.LongTensor(diagnosis_codes)
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
            mask_code = torch.Tensor(mask_code).unsqueeze(3)
        features = self.feature_encoder(diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))

        self_weight = self.get_self_attention(features, quiryes, mask_mult)
        time_weight = self.time_encoder(seq_time_step, quiryes, options, mask_mult)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight


class TransformerTimeEmb(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerTimeEmb, self).__init__()
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = EncoderNew(options['n_diagnosis_codes'] + 1, 51, num_layers=options['layer'])
        self.self_layer = torch.nn.Linear(256, 1)
        self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill_(mask, -np.inf), dim=1)
        return attention

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(rnn_tools.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final, mask_code = rnn_tools.pad_matrix_new(seq_dignosis_codes,
                                                                                        batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.LongTensor(diagnosis_codes).cuda()
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
            mask_code = torch.Tensor(mask_code).unsqueeze(3).cuda()
        else:
            diagnosis_codes = torch.LongTensor(diagnosis_codes)
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
            mask_code = torch.Tensor(mask_code).unsqueeze(3)
        features = self.feature_encoder(diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))

        self_weight = self.get_self_attention(features, quiryes, mask_mult)
 
        total_weight = self_weight
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight


class TransformerSelf(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerSelf, self).__init__()
        self.feature_encoder = EncoderPure(options['n_diagnosis_codes'] + 1, 51, num_layers=options['layer'])
        self.self_layer = torch.nn.Linear(256, 1)
        self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill_(mask, -np.inf), dim=1)
        return attention

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(rnn_tools.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final, mask_code = rnn_tools.pad_matrix_new(seq_dignosis_codes,
                                                                                        batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.LongTensor(diagnosis_codes).cuda()
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
            mask_code = torch.Tensor(mask_code).unsqueeze(3).cuda()
        else:
            diagnosis_codes = torch.LongTensor(diagnosis_codes)
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
            mask_code = torch.Tensor(mask_code).unsqueeze(3)
        features = self.feature_encoder(diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))

        self_weight = self.get_self_attention(features, quiryes, mask_mult)
        total_weight = self_weight
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, self_weight


class TransformerFinal(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerFinal, self).__init__()
        self.feature_encoder = EncoderPure(options['n_diagnosis_codes'] + 1, 51, num_layers=options['layer'])
        self.self_layer = torch.nn.Linear(256, 1)
        self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(rnn_tools.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final, mask_code = rnn_tools.pad_matrix_new(seq_dignosis_codes,
                                                                                        batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.LongTensor(diagnosis_codes).cuda()
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
            mask_code = torch.Tensor(mask_code).unsqueeze(3).cuda()
        else:
            diagnosis_codes = torch.LongTensor(diagnosis_codes)
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
            mask_code = torch.Tensor(mask_code).unsqueeze(3)
        features = self.feature_encoder(diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1)

        predictions = self.classify_layer(final_statues)
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, None

class TransformerEval(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerEval, self).__init__()
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = EncoderEval(options['n_diagnosis_codes'] + 1, 51, num_layers=options['layer'])
        self.self_layer = torch.nn.Linear(256, 1)
        self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill_(mask, -np.inf), dim=1)
        return attention

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(rnn_tools.pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final, mask_code = rnn_tools.pad_matrix_new(seq_dignosis_codes,
                                                                                        batch_labels, options)
        if options['use_gpu']:
            diagnosis_codes = torch.LongTensor(diagnosis_codes).cuda()
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
            mask_code = torch.Tensor(mask_code).unsqueeze(3).cuda()
        else:
            diagnosis_codes = torch.LongTensor(diagnosis_codes)
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
            mask_code = torch.Tensor(mask_code).unsqueeze(3)
        features, attention_local = self.feature_encoder(diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))

        self_weight = self.get_self_attention(features, quiryes, mask_mult)
        time_weight = self.time_encoder(seq_time_step, quiryes, options, mask_mult)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features)
        attention_local = attention_local.view(-1, 4, attention_local.size(1), attention_local.size(2))
        labels = torch.LongTensor(labels)
        if options['use_gpu']:
            labels = labels.cuda()
        return predictions, labels, (time_weight, self_weight, attention_weight, total_weight, attention_local)
