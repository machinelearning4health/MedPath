

import pickle
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from modeling.untils import adjust_input


def load_data(training_file, validation_file, testing_file):
    train = np.array(pickle.load(open(training_file, 'rb')))
    validate = np.array(pickle.load(open(validation_file, 'rb')))
    test = np.array(pickle.load(open(testing_file, 'rb')))
    return train, validate, test


def pad_matrix(seq_diagnosis_codes, seq_labels, n_diagnosis_codes, max_num_pervisit):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    n_samples = len(seq_diagnosis_codes)
    maxlen = np.max(lengths)

    batch_diagnosis_codes = np.zeros((maxlen, n_samples, max_num_pervisit), dtype=np.float32)
    batch_mask = np.zeros((maxlen, n_samples), dtype=np.float32)

    for idx, c in enumerate(seq_diagnosis_codes):
        for x, subseq in zip(batch_diagnosis_codes[:, idx, :], c[:]):
            for i in range(len(x)):
                if i<len(subseq):
                    x[i] = subseq[i]
                else:
                    x[i] = n_diagnosis_codes
    for i in range(n_samples):
        max_visit = lengths[i] - 1
        batch_mask[max_visit, i] = 1


    return batch_diagnosis_codes, batch_mask


def pad_matrix_time(seq_diagnosis_codes, seq_times, n_diagnosis_codes):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    lengths_code = []
    for seq in seq_diagnosis_codes:
        for code_set in seq:
            lengths_code.append(len(code_set))
    lengths_code = np.array(lengths_code)
    maxcode = np.max(lengths_code)
    n_samples = len(seq_diagnosis_codes)
    maxlen = np.max(lengths)

    batch_diagnosis_codes = np.zeros((n_samples, maxlen, maxcode), dtype=np.int64) + n_diagnosis_codes
    batch_mask = np.zeros((n_samples, maxlen, maxcode), dtype=np.float32) + 1e+20
    batch_mask_final = np.zeros((n_samples, maxlen), dtype=np.float32)

    for bid, seq in enumerate(seq_diagnosis_codes):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_diagnosis_codes[bid, pid, tid] = code
                batch_mask[bid, pid, tid] = 0

    for k in range(len(seq_times)):
        while len(seq_times[k]) < maxlen:
            seq_times[k].append(100000)

    for i in range(n_samples):
        max_visit = lengths[i] - 1
        batch_mask_final[i, max_visit] = 1

    batch_times = np.array(seq_times, dtype=np.float32)

    return batch_diagnosis_codes, batch_mask, batch_times, batch_mask_final


def pad_matrix_time_time(seq_diagnosis_codes, seq_labels, seq_times, options, ahead):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    lengths_code = []
    for seq in seq_diagnosis_codes:
        for code_set in seq:
            lengths_code.append(len(code_set))
    lengths_code = np.array(lengths_code)
    maxcode = np.max(lengths_code)
    n_samples = len(seq_diagnosis_codes)
    n_diagnosis_codes = options['max_code']
    maxlen = min(np.max(lengths), ahead)

    batch_diagnosis_codes = np.zeros((n_samples, maxlen, maxcode), dtype=np.int64) + options['n_diagnosis_codes']
    batch_mask = np.zeros((n_samples, maxlen, maxcode), dtype=np.float32) + 1e+20
    batch_mask_final = np.zeros((n_samples, maxlen), dtype=np.float32)

    for bid, seq in enumerate(seq_diagnosis_codes):
        for pid, subseq in enumerate(seq[0:ahead]):
            for tid, code in enumerate(subseq):
                batch_diagnosis_codes[bid, pid, tid] = code
                batch_mask[bid, pid, tid] = 0

    seq_times = seq_times.tolist()
    for k in range(len(seq_times)):
        while len(seq_times[k]) < maxlen:
            seq_times[k].append(100000)

    for i in range(n_samples):
        max_visit = min(lengths[i], ahead) - 1
        batch_mask_final[i, max_visit] = 1

    batch_labels = np.array(seq_labels, dtype=np.int64)
    batch_times = np.array(seq_times, dtype=np.float32)
    batch_times = batch_times[:, 0:ahead]
    return batch_diagnosis_codes, batch_labels, batch_mask, batch_times, batch_mask_final


def pad_matrix_retainEx(seq_diagnosis_codes, seq_labels, seq_times, n_diagnosis_codes, maxlen, maxcode):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    lengths_code = []
    for seq in seq_diagnosis_codes:
        for code_set in seq:
            lengths_code.append(len(code_set))
    lengths_code = np.array(lengths_code)
    maxcode = np.max(lengths_code)
    n_samples = len(seq_diagnosis_codes)

    batch_diagnosis_codes = np.zeros((n_samples, maxlen, maxcode), dtype=np.int64) + n_diagnosis_codes
    batch_mask = np.zeros((n_samples, maxlen, maxcode), dtype=np.float32)
    batch_mask_final = np.zeros((n_samples, maxlen), dtype=np.float32)

    for bid, seq in enumerate(seq_diagnosis_codes):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_diagnosis_codes[bid, pid, tid] = code
                batch_mask[bid, pid, tid] = 1

    for k in range(len(seq_times)):
        while len(seq_times[k]) < maxlen:
            seq_times[k].append(100000)

    for i in range(n_samples):
        max_visit = lengths[i] - 1
        batch_mask_final[i, max_visit] = 1

    batch_times = np.array(seq_times, dtype=np.float32)

    return batch_diagnosis_codes, batch_mask, batch_times, batch_mask_final


def pad_matrix_retainEx_time(seq_diagnosis_codes, seq_labels, seq_times, options, ahead):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    lengths_code = []
    for seq in seq_diagnosis_codes:
        for code_set in seq:
            lengths_code.append(len(code_set))
    lengths_code = np.array(lengths_code)
    maxcode = np.max(lengths_code)
    n_samples = len(seq_diagnosis_codes)
    n_diagnosis_codes = options['max_code']
    maxlen = min(np.max(lengths), ahead)

    batch_diagnosis_codes = np.zeros((n_samples, maxlen, maxcode), dtype=np.int64) + options['n_diagnosis_codes']
    batch_mask = np.zeros((n_samples, maxlen, maxcode), dtype=np.float32)
    batch_mask_final = np.zeros((n_samples, maxlen), dtype=np.float32)

    for bid, seq in enumerate(seq_diagnosis_codes):
        for pid, subseq in enumerate(seq[0:ahead]):
            for tid, code in enumerate(subseq):
                batch_diagnosis_codes[bid, pid, tid] = code
                batch_mask[bid, pid, tid] = 1

    seq_times = seq_times.tolist()
    for k in range(len(seq_times)):
        while len(seq_times[k]) < maxlen:
            seq_times[k].append(100000)

    for i in range(n_samples):
        max_visit = min(lengths[i], ahead) - 1
        batch_mask_final[i, max_visit] = 1

    batch_labels = np.array(seq_labels, dtype=np.int64)
    batch_times = np.array(seq_times, dtype=np.float32)
    batch_times = batch_times[:, 0:ahead]

    return batch_diagnosis_codes, batch_labels, batch_mask, batch_times, batch_mask_final


def pad_matrix_con(seq_diagnosis_codes, seq_labels, seq_times, options):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    n_samples = len(seq_diagnosis_codes)
    n_diagnosis_codes = options['n_diagnosis_codes']
    maxlen = np.max(lengths)
    lengths_code = []
    for seq in seq_diagnosis_codes:
        for code_set in seq:
            lengths_code.append(len(code_set))
    lengths_code = np.array(lengths_code)
    maxcode = np.max(lengths_code)

    batch_diagnosis_codes = np.zeros((n_samples, maxlen, maxcode), dtype=np.int64)
    batch_mask = np.zeros((n_samples, maxlen), dtype=np.float32)
    batch_mask_code = np.zeros((n_samples, maxlen, maxcode), dtype=np.float32)

    for bid, seq in enumerate(seq_diagnosis_codes):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_diagnosis_codes[bid, pid, tid] = code
                batch_mask_code[bid, pid, tid] = 1
                batch_mask[bid, pid,] = 1

    seq_times = seq_times.tolist()
    for k in range(len(seq_times)):
        while len(seq_times[k]) < maxlen:
            seq_times[k].append(100000)

    batch_labels = np.array(seq_labels, dtype=np.int64)
    batch_times = np.array(seq_times, dtype=np.float32)

    return batch_diagnosis_codes, batch_labels, batch_mask, batch_times, batch_mask_code


def pad_time(seq_time_step, options):
    lengths = np.array([len(seq) for seq in seq_time_step])
    maxlen = np.max(lengths)
    for k in range(len(seq_time_step)):
        while len(seq_time_step[k]) < maxlen:
            seq_time_step[k].append(100000)

    return seq_time_step


def pad_matrix_mine(seq_diagnosis_codes, n_diagnosis_codes, max_num_pervisit):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    n_samples = len(seq_diagnosis_codes)
    maxlen = np.max(lengths)

    batch_diagnosis_codes = np.zeros((maxlen, n_samples, max_num_pervisit), dtype=np.float32)
    batch_mask = np.zeros((maxlen, n_samples), dtype=np.float32)
    batch_mask_final = np.zeros((maxlen, n_samples), dtype=np.float32)

    for idx, c in enumerate(seq_diagnosis_codes):
        for x, subseq in zip(batch_diagnosis_codes[:, idx, :], c[:]):
            for i in range(len(x)):
                if i < len(subseq):
                    x[i] = subseq[i]
                else:
                    x[i] = n_diagnosis_codes

    for i in range(n_samples):
        batch_mask[0:lengths[i], i] = 1
        max_visit = lengths[i] - 1
        batch_mask_final[max_visit, i] = 1


    return batch_diagnosis_codes, batch_mask, batch_mask_final


def pad_matrix_new(seq_diagnosis_codes, n_diagnosis_codes):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    n_samples = len(seq_diagnosis_codes)
    maxlen = np.max(lengths)
    lengths_code = []
    for seq in seq_diagnosis_codes:
        for code_set in seq:
            lengths_code.append(len(code_set))
    lengths_code = np.array(lengths_code)
    maxcode = np.max(lengths_code)

    batch_diagnosis_codes = np.zeros((n_samples, maxlen, maxcode), dtype=np.int64) + n_diagnosis_codes
    batch_mask = np.zeros((n_samples, maxlen), dtype=np.float32)
    batch_mask_code = np.zeros((n_samples, maxlen, maxcode), dtype=np.float32)
    batch_mask_final = np.zeros((n_samples, maxlen), dtype=np.float32)

    for bid, seq in enumerate(seq_diagnosis_codes):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_diagnosis_codes[bid, pid, tid] = code
                batch_mask_code[bid, pid, tid] = 1

    for i in range(n_samples):
        batch_mask[i, 0:lengths[i] - 1] = 1
        max_visit = lengths[i] - 1
        batch_mask_final[i, max_visit] = 1


    return batch_diagnosis_codes, batch_mask, batch_mask_final, batch_mask_code


def calculate_cost(model, data, options, loss_function=F.cross_entropy):
    batch_size = options['batch_size']
    n_batches = int(np.ceil(float(len(data[0])) / float(batch_size)))
    cost_sum = 0.0
    model.eval()
    for index in range(n_batches):
        batch_diagnosis_codes = data[0][batch_size * index: batch_size * (index + 1)]
        for ind in range(len(batch_diagnosis_codes)):
            if len(batch_diagnosis_codes[ind]) > 50:
                batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-50:]
        batch_labels = data[1][batch_size * index: batch_size * (index + 1)]
        t_diagnosis_codes, t_labels, t_mask = pad_matrix(batch_diagnosis_codes, batch_labels, options)

        if options['use_gpu']:
            t_diagnosis_codes = Variable(torch.FloatTensor(t_diagnosis_codes).cuda())
            t_labels = Variable(torch.LongTensor(t_labels).cuda())
            t_mask = Variable(torch.FloatTensor(t_mask).cuda())
        else:
            t_diagnosis_codes = Variable(torch.FloatTensor(t_diagnosis_codes))
            t_labels = Variable(torch.LongTensor(t_labels))
            t_mask = Variable(torch.FloatTensor(t_mask))

        logit = model(t_diagnosis_codes, t_mask)
        loss = loss_function(logit, t_labels)
        cost_sum += loss.cpu().data.numpy()
    model.train()
    return cost_sum / n_batches


def calculate_cost_rnn_attetntion(model, data, options, loss_function=F.cross_entropy):
    batch_size = options['batch_size']
    n_batches = int(np.ceil(float(len(data[0])) / float(batch_size)))
    cost_sum = 0.0
    model.eval()
    for index in range(n_batches):
        batch_diagnosis_codes = data[0][batch_size * index: batch_size * (index + 1)]
        for ind in range(len(batch_diagnosis_codes)):
            if len(batch_diagnosis_codes[ind]) > 50:
                batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-50:]
        batch_labels = data[1][batch_size * index: batch_size * (index + 1)]
        t_diagnosis_codes, t_labels, t_mask, t_mask_final = pad_matrix_mine(batch_diagnosis_codes, batch_labels,
                                                                            options)

        if options['use_gpu']:
            t_diagnosis_codes = Variable(torch.FloatTensor(t_diagnosis_codes).cuda())
            t_labels = Variable(torch.LongTensor(t_labels).cuda())
            t_mask = Variable(torch.FloatTensor(t_mask).cuda())
        else:
            t_diagnosis_codes = Variable(torch.FloatTensor(t_diagnosis_codes))
            t_labels = Variable(torch.LongTensor(t_labels))
            t_mask = Variable(torch.FloatTensor(t_mask))

        logit = model(t_diagnosis_codes, t_mask)
        loss = loss_function(logit, t_labels)
        cost_sum += loss.cpu().data.numpy()
    model.train()
    return cost_sum / n_batches


def calculate_cost_mine(model, data, options, loss_function=F.cross_entropy):
    model.eval()
    batch_size = options['batch_size']
    n_batches = int(np.ceil(float(len(data[0])) / float(batch_size)))
    cost_sum = 0.0

    for index in range(n_batches):
        batch_diagnosis_codes = data[0][batch_size * index: batch_size * (index + 1)]
        batch_time_step = data[2][batch_size * index: batch_size * (index + 1)]
        for ind in range(len(batch_diagnosis_codes)):
            if len(batch_diagnosis_codes[ind]) > 50:
                batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-50:]
                batch_time_step[ind] = batch_time_step[ind][-50:]

        batch_labels = data[1][batch_size * index: batch_size * (index + 1)]
        lengths = np.array([len(seq) for seq in batch_diagnosis_codes])
        maxlen = np.max(lengths)
        logit, labels = model(batch_diagnosis_codes, batch_time_step, batch_labels, options, maxlen)
        loss = loss_function(logit, labels)
        cost_sum += loss.cpu().data.numpy()
    model.train()
    return cost_sum / n_batches


def calculate_cost_tran(model, data, options, max_len, loss_function=F.cross_entropy):
    model.eval()
    batch_size = options['batch_size']
    n_batches = int(np.ceil(float(len(data[0])) / float(batch_size)))
    cost_sum = 0.0

    for index in range(n_batches):
        batch_diagnosis_codes = data[0][batch_size * index: batch_size * (index + 1)]
        batch_time_step = data[2][batch_size * index: batch_size * (index + 1)]
        batch_diagnosis_codes, batch_time_step = adjust_input(batch_diagnosis_codes, batch_time_step, max_len,
                                                              options['n_diagnosis_codes'])
        batch_labels = data[1][batch_size * index: batch_size * (index + 1)]
        lengths = np.array([len(seq) for seq in batch_diagnosis_codes])
        maxlen = np.max(lengths)
        logit, labels, self_attention = model(batch_diagnosis_codes, batch_time_step, batch_labels, options, maxlen)
        loss = loss_function(logit, labels)
        cost_sum += loss.cpu().data.numpy()
    model.train()
    return cost_sum / n_batches


def calculate_cost_time(model, data, options, max_len, loss_function=F.cross_entropy):
    batch_size = options['batch_size']
    n_batches = int(np.ceil(float(len(data[0])) / float(batch_size)))
    cost_sum = 0.0
    model.eval()
    for index in range(n_batches):
        batch_diagnosis_codes = data[0][batch_size * index: batch_size * (index + 1)]
        batch_labels = data[1][batch_size * index: batch_size * (index + 1)]
        batch_time_step = data[2][batch_size * index: batch_size * (index + 1)]
        for ind in range(len(batch_diagnosis_codes)):
            if len(batch_diagnosis_codes[ind]) > 50:
                batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-50:]
                batch_time_step[ind] = batch_time_step[ind][-50:]
        t_diagnosis_codes, t_labels, t_mask, t_time, t_mask_final = pad_matrix_time(batch_diagnosis_codes, batch_labels,
                                                                                    batch_time_step,
                                                                                    options)

        t_diagnosis_codes = Variable(torch.LongTensor(t_diagnosis_codes).cuda())
        t_labels = Variable(torch.LongTensor(t_labels).cuda())
        t_mask = Variable(torch.FloatTensor(t_mask).cuda())
        t_mask_final = Variable(torch.FloatTensor(t_mask_final).cuda())
        t_time = Variable(torch.FloatTensor(t_time).cuda())

        logit = model([t_diagnosis_codes, t_mask, t_time, t_mask_final], None)
        loss = loss_function(logit, t_labels)
        cost_sum += loss.cpu().data.numpy()
    model.train()
    return cost_sum / n_batches


def calculate_cost_retainEx(model, data, options, max_len, loss_function=F.cross_entropy):
    batch_size = options['batch_size']
    n_batches = int(np.ceil(float(len(data[0])) / float(batch_size)))
    cost_sum = 0.0
    model.eval()
    for index in range(n_batches):
        batch_diagnosis_codes = data[0][batch_size * index: batch_size * (index + 1)]
        batch_labels = data[1][batch_size * index: batch_size * (index + 1)]
        batch_time_step = data[2][batch_size * index: batch_size * (index + 1)]
        for ind in range(len(batch_diagnosis_codes)):
            if len(batch_diagnosis_codes[ind]) > 50:
                batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-50:]
                batch_time_step[ind] = batch_time_step[ind][-50:]
        t_diagnosis_codes, t_labels, t_mask, t_time, t_mask_final = pad_matrix_retainEx(batch_diagnosis_codes,
                                                                                        batch_labels, batch_time_step,
                                                                                        options)

        t_diagnosis_codes = Variable(torch.LongTensor(t_diagnosis_codes).cuda())
        t_labels = Variable(torch.LongTensor(t_labels).cuda())
        t_mask = Variable(torch.FloatTensor(t_mask).cuda())
        t_mask_final = Variable(torch.FloatTensor(t_mask_final).cuda())
        t_time = Variable(torch.FloatTensor(t_time).cuda())

        logit = model([t_diagnosis_codes, t_mask, t_time, t_mask_final], None)
        loss = loss_function(logit, t_labels)
        cost_sum += loss.cpu().data.numpy()
    model.train()
    return cost_sum / n_batches


def calculate_cost_con_2(model, data, options, max_len, loss_function=F.cross_entropy):
    batch_size = options['batch_size']
    n_batches = int(np.ceil(float(len(data[0])) / float(batch_size)))
    cost_sum = 0.0
    model.eval()
    for index in range(n_batches):
        batch_diagnosis_codes = data[0][batch_size * index: batch_size * (index + 1)]
        batch_time_step = data[2][batch_size * index: batch_size * (index + 1)]
        batch_labels = data[1][batch_size * index: batch_size * (index + 1)]
        for ind in range(len(batch_diagnosis_codes)):
            if len(batch_diagnosis_codes[ind]) > 50:
                batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-50:]
                batch_time_step[ind] = batch_time_step[ind][-50:]
        lengths = np.array([len(seq) for seq in batch_diagnosis_codes])
        maxlen = np.max(lengths)
        t_diagnosis_codes, t_labels, t_mask, t_time, t_mask_code = pad_matrix_con(batch_diagnosis_codes,
                                                                                  batch_labels,
                                                                                  batch_time_step, options)
        t_diagnosis_codes = Variable(torch.LongTensor(t_diagnosis_codes).cuda())
        t_labels = Variable(torch.LongTensor(t_labels).cuda())
        t_mask = Variable(torch.FloatTensor(t_mask).cuda())
        t_mask_code = Variable(torch.FloatTensor(t_mask_code).cuda())
        t_time = Variable(torch.FloatTensor(t_time).cuda())

        logit, decov_loss = model(t_diagnosis_codes, t_time, t_mask_code)
        loss = loss_function(logit, t_labels)
        cost_sum += loss.cpu().data.numpy()
    model.train()
    return cost_sum / n_batches


def calculate_cost_con(model, data, options, max_len, loss_function=F.cross_entropy):
    batch_size = options['batch_size']
    n_batches = int(np.ceil(float(len(data[0])) / float(batch_size)))
    cost_sum = 0.0
    model.eval()
    for index in range(n_batches):
        batch_diagnosis_codes = data[0][batch_size * index: batch_size * (index + 1)]
        batch_time_step = data[2][batch_size * index: batch_size * (index + 1)]
        batch_labels = data[1][batch_size * index: batch_size * (index + 1)]
        for ind in range(len(batch_diagnosis_codes)):
            if len(batch_diagnosis_codes[ind]) > 50:
                batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-50:]
                batch_time_step[ind] = batch_time_step[ind][-50:]
        lengths = np.array([len(seq) for seq in batch_diagnosis_codes])
        maxlen = np.max(lengths)
        t_diagnosis_codes, t_labels, t_mask, t_time, t_mask_code = pad_matrix_con(batch_diagnosis_codes,
                                                                                  batch_labels,
                                                                                  batch_time_step, options)
        t_diagnosis_codes = Variable(torch.LongTensor(t_diagnosis_codes).cuda())
        t_labels = Variable(torch.LongTensor(t_labels).cuda())
        t_mask = Variable(torch.FloatTensor(t_mask).cuda())
        t_mask_code = Variable(torch.FloatTensor(t_mask_code).cuda())
        t_time = Variable(torch.FloatTensor(t_time).cuda())

        logit = model(t_diagnosis_codes, t_time, t_mask_code)
        loss = loss_function(logit, t_labels)
        cost_sum += loss.cpu().data.numpy()
    model.train()
    return cost_sum / n_batches
