import math
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from DB_generation import *
import time
import os
import sys
import argparse
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
import random
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/exp')
from torch.cuda.amp import autocast, GradScaler
import pandas as pd




Words1 = ['Pad', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn' ,'H', 'Cu',
          'Mn', '-', '=', '#', '~', '?', '?5', '.', '>','Pd', 'Li', 'Sn', 'Mo', 'Co', 'Cs', 'As']
Words2 = ['Pad', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn' ,'H', 'Cu',
          'Mn', '-', '=', '#', '~', '?', '?5', '.', 'sos', 'eos', 'Pd', 'Li', 'Sn', 'Mo', 'Co', 'Cs', 'As']

src_vocab = {'Pad': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Si': 6, 'P': 7, 'Cl': 8, 'Br': 9, 'Mg': 10,
             'Na': 11, 'Ca': 12, 'Fe': 13, 'Al': 14, 'I': 15, 'B': 16, 'K': 17, 'Se': 18, 'Zn': 19, 'H': 20,
             'Cu': 21, 'Mn': 22, '-': 23, '=': 24, '#': 25, '~': 26, '.': 27, '>': 28, '?1': 29, '?2': 30,
             '?3': 31, '?4': 32, '?5': 33, '?6': 34, '?7': 35, '?8': 36, '?9': 37, '?10': 38, '?11': 39,
             '?12': 40, '?13': 41, '?14': 42, '?15': 43, '?16': 44, '?17': 45, '?18': 46, '?19': 47, '?20': 48,
             '?21': 49, '?22': 50, '?23': 51, '?24': 52, '?25': 53, '?26': 54, '?27': 55, '?28': 56, '?29': 57,
             '?30': 58, '?31': 59, '?32': 60, '?33': 61, '?34': 62, '?35': 63, '?36': 64, '?37': 65, '?38': 66,
             '?39': 67, '?40': 68, '?41': 69, '?42': 70, '?43': 71, '?44': 72, '?45': 73, '?46': 74, '?47': 75,
             '?48': 76, '?49': 77, '?50': 78, 'Pd': 79, 'Li': 80, 'Sn': 81, 'Mo': 82, 'Co': 83, 'Cs': 84, 'As': 85}
src_vocab_size = len(src_vocab)

tgt_vocab = {'Pad': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Si': 6, 'P': 7, 'Cl': 8, 'Br': 9, 'Mg': 10,
             'Na': 11, 'Ca': 12, 'Fe': 13, 'Al': 14, 'I': 15, 'B': 16, 'K': 17, 'Se': 18, 'Zn': 19, 'H': 20,
             'Cu': 21, 'Mn': 22, '-': 23, '=': 24, '#': 25, '~': 26, '.': 27, '?1': 28, '?2': 29, '?3': 30,
             '?4': 31, '?5': 32, '?6': 33, '?7': 34, '?8': 35, '?9': 36, '?10': 37, '?11': 38, '?12': 39,
             '?13': 40, '?14': 41, '?15': 42, '?16': 43, '?17': 44, '?18': 45, '?19': 46, '?20': 47, '?21': 48,
             '?22': 49, '?23': 50, '?24': 51, '?25': 52, '?26': 53, '?27': 54, '?28': 55, '?29': 56, '?30': 57,
             '?31': 58, '?32': 59, '?33': 60, '?34': 61, '?35': 62, '?36': 63, '?37': 64, '?38': 65, '?39': 66,
             '?40': 67, '?41': 68, '?42': 69, '?43': 70, '?44': 71, '?45': 72, '?46': 73, '?47': 74, '?48': 75,
             '?49': 76, '?50': 77,  'sos': 78, 'eos': 79, 'Pd': 80, 'Li': 81, 'Sn': 82, 'Mo': 83, 'Co': 84, 'Cs': 85, 'As': 86}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 150 # enc_input max sequence length
tgt_len = 150 # dec_input(=dec_output) max sequence length

# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention


def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]  # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]  # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True. 
        scores.masked_fill_(attn_mask, np.half(-1e9))  # Fills elements of self tensor with value where mask is True. 
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda(args.local_rank)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda(args.local_rank)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda(args.local_rank)  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda(args.local_rank) # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda(args.local_rank)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).cuda(args.local_rank) # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda(args.local_rank)
        self.decoder = Decoder().cuda(args.local_rank)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda(args.local_rank)

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    #print(enc_outputs.shape)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = True
    next_symbol = start_symbol
    t_n = 0
    n = 5  
    while terminal == True and n > 0:
        n = n - 1
        t_n = t_n + 1
        dec_input = torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype).cuda(args.local_rank)],-1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab['eos']:
            terminal = False

    return dec_input

def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", help="local device id on current node", type=int)
    args = parser.parse_args()

    if torch.cuda.is_available():
        logging.warning("Cuda is available!")
        if torch.cuda.device_count() > 1:
            logging.warning((f"Find {torch.cuda.device_count()} GPUs!"))
        else:
            logging.warning("Too few GPUs!")
            sys.exit(0)
    else:
        logging.warning("Cuda is not available! Exit!")
        sys.exit(0)

    n_gpus = 2
    torch.distributed.init_process_group("nccl", world_size=n_gpus, rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)
    model = Transformer()
    model = nn.parallel.DistributedDataParallel(model.cuda(args.local_rank), device_ids=[args.local_rank])  # DistributedDataParallel

    data_t = pd.read_csv(r"/home/user06/model/data/MIT_separate_50/train2.0.csv")
    Sentences = []
    # len(data_t.index)
    for i in range(len(data_t.index)):
        sentence = []
        sentence.append(data_t["enc_input"][i])
        sentence.append(data_t["dec_input"][i])
        sentence.append(data_t["dec_output"][i])
        Sentences.append(sentence)
    random.shuffle(Sentences)
    enc_inputs, dec_inputs, dec_outputs = make_data(Sentences)
    train_sampler = DistributedSampler(MyDataSet(enc_inputs, dec_inputs, dec_outputs))
    batch_size = 64
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=batch_size, sampler=train_sampler)


    data_v = pd.read_csv(r"/home/user06/model/data/MIT_separate_50/val.csv")
    Sentences_v = []
    for i in range(len(data_v.index)):
        sentence_v = []
        sentence_v.append(data_v["enc_input"][i])
        sentence_v.append(data_v["dec_input"][i])
        sentence_v.append(data_v["dec_output"][i])
        Sentences_v.append(sentence_v)
    random.shuffle(Sentences_v)
    # Sentences_v_input = Sentences_v[:300]
    enc_inputs_v, dec_inputs_v, dec_outputs_v = make_data(Sentences_v)
    val_sampler = DistributedSampler(MyDataSet(enc_inputs_v, dec_inputs_v, dec_outputs_v))
    batch_size = 2
    loader_v = Data.DataLoader(MyDataSet(enc_inputs_v, dec_inputs_v, dec_outputs_v), batch_size=batch_size, sampler=val_sampler)


    criterion = nn.CrossEntropyLoss(ignore_index=0)
    #  optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-07, weight_decay=0, foreach=None)
    #  optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.0001)
    #  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda step: rate(step, d_model, factor=3000, warmup=10000),
                                                     last_epoch=-1)

    start_time = time.time()
    
  
    scaler = GradScaler()
    
    sum_step = 0
    for epoch in range(500):
        print("---------- {} start----------".format(epoch+1))
        train_sampler.set_epoch(epoch)  
        sum_loss = 0
        step = 0
        for enc_inputs, dec_inputs, dec_outputs in loader:
          '''
          enc_inputs: [batch_size, src_len]
          dec_inputs: [batch_size, tgt_len]
          dec_outputs: [batch_size, tgt_len]
          '''
          optimizer.zero_grad()
          
          sum_step = sum_step + 1
          step = step + 1
          enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(args.local_rank), dec_inputs.cuda(args.local_rank), dec_outputs.cuda(args.local_rank)

          with autocast(enabled=True):
              outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
              loss = criterion(outputs, dec_outputs.view(-1))

          # loss = criterion(outputs, dec_outputs.view(-1))
          writer.add_scalar('loss', loss.item(), global_step=epoch + 1)
          
          scaler.scale(loss).backward()
          scaler.unscale_(optimizer)
          # torch.nn.utils.clip_grad_norm(model.parameters(), max_norm)
          scaler.step(optimizer)
          scaler.update()
          lr_scheduler.step()
          
          sum_loss = sum_loss + loss.item()
          writer.add_scalar('lr_scheduler', optimizer.state_dict()["param_groups"][0]["lr"], global_step=epoch + 1)
          if sum_step % 5 == 0:
              print('Step:', '%08d' % (sum_step) , 'Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.item()))
              print('learning rate:',optimizer.state_dict()["param_groups"][0]["lr"])

        print(time.time() - start_time)
        sum_loss = sum_loss / step
        print("total_loss:{}".format(sum_loss))
        writer.add_scalar('total_loss', sum_loss, global_step=epoch + 1)
        start_time = time.time()
        

        loss_v_total = 0
        step_v = 0
        for enc_inputs_v, dec_inputs_v, dec_outputs_v in loader_v:
            step_v = step_v + 1
            enc_inputs_v, dec_inputs_v, dec_outputs_v = enc_inputs_v.cuda(args.local_rank), dec_inputs_v.cuda(args.local_rank), dec_outputs_v.cuda(args.local_rank)
            outputs_v, enc_self_attns_v, dec_self_attns_v, dec_enc_attns_v = model(enc_inputs_v, dec_inputs_v)
            loss_v = criterion(outputs_v, dec_outputs_v.view(-1))
            loss_v_total = loss_v_total + loss_v.item()
            writer.add_scalar('loss_val', loss_v, global_step=epoch + 1)
            
        loss_avg = loss_v_total / step_v
        print("loss_val:{}".format(loss_avg))

       

        if (epoch+1) % 5 == 0 and args.local_rank == 0:
            print('epoch:', epoch+1)
            print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
            print('learning rate:', lr_scheduler.get_last_lr()[0])
            checkpoint = {
                "model": model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'lr_schedule': lr_scheduler.state_dict()
            }
            save_path = './model'
            filename = 'ckpt_epoch_{}.pth'.format(epoch+1)
            torch.save(checkpoint, os.path.join(save_path, filename))
            print("model save")

