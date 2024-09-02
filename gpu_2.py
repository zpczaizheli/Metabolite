import math
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
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
from torch.nn import Linear, ReLU, Sigmoid
from torch.nn.init import kaiming_uniform_, xavier_uniform_


Words1 = ['Pad', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn' ,'H', 'Cu',
          'Mn', '-', '=', '#', '~', '?', '?5', '.', '>','Pd', 'Li', 'Sn', 'Mo', 'Co', 'Cs', 'As']
Words2 = ['Pad', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn' ,'H', 'Cu',
          'Mn', '-', '=', '#', '~', '?', '?5', '.', 'sos', 'eos', 'Pd', 'Li', 'Sn', 'Mo', 'Co', 'Cs', 'As']

src_vocab = {'Pad': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Si': 6, 'P': 7, 'Cl': 8, 'Br': 9, 'Mg': 10,
             'Na': 11, 'Ca': 12, 'Fe': 13, 'Al': 14, 'I': 15, 'B': 16, 'K': 17, 'Se': 18, 'Zn': 19, 'H': 20,
             'Cu': 21, 'Mn': 22, '-': 23, '=': 24, '#': 25, '~': 26, 'Pd': 27, 'Li': 28, 'Sn': 29, 'Mo': 30, 'Co': 31, 'Cs': 32, 'As': 33}
src_vocab_size = len(src_vocab)

tgt_vocab = {'Pad': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Si': 6, 'P': 7, 'Cl': 8, 'Br': 9, 'Mg': 10,
             'Na': 11, 'Ca': 12, 'Fe': 13, 'Al': 14, 'I': 15, 'B': 16, 'K': 17, 'Se': 18, 'Zn': 19, 'H': 20,
             'Cu': 21, 'Mn': 22, '-': 23, '=': 24, '#': 25, '~': 26, 'sos': 27, 'eos': 28, 'Pd': 29, 'Li': 30, 'Sn': 31, 'Mo': 32, 'Co': 33, 'Cs': 34, 'As': 35}
tgt_vocab_size = len(tgt_vocab)
idx2word = {i: w for i, w in enumerate(tgt_vocab)}

# degree coding New
degree_vocab = {'pad': 7, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}
degree_vocab_size = len(degree_vocab)

# distance coding New
distance_vocab = {'-2': 82, '-1': 81, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
                  '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20,
                  '21': 21, '22': 22, '23': 23, '24': 24, '25': 25, '26': 26, '27': 27, '28': 28, '29': 29, '30': 30,
                  '31': 31, '32': 32, '33': 33, '34': 34, '35': 35, '36': 36, '37': 37, '38': 38, '39': 39, '40': 40,
                  '41': 41, '42': 42, '43': 43, '44': 44, '45': 45, '46': 46, '47': 47, '48': 48, '49': 49, '50': 50,
                  '51': 51, '52': 52, '53': 53, '54': 54, '55': 55, '56': 56, '57': 57, '58': 58, '59': 59, '60': 60,
                  '61': 61, '62': 62, '63': 63, '64': 64, '65': 65, '66': 66, '67': 67, '68': 68, '69': 69, '70': 70,
                  '71': 71, '72': 72, '73': 73, '74': 74, '75': 75, '76': 76, '77': 77, '78': 78, '79': 79, '80': 80}
distance_vocab_size = len(distance_vocab)

src_len = 300 
tgt_len = 300 


d_model = 64 
d_ff = 512
d_k = d_v = 16  
n_layers = 6 
n_heads = 4  

def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs, degree_s, MD, MA= [], [], [], [], [], []
    for i in range(len(sentences)):
        print(i)
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]  
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]  
        dec_s1 = sentences[i][2].split()
        dec_s2 = ["Pad" if i == "pad" else i for i in dec_s1]
        dec_output = [[tgt_vocab[n] for n in dec_s2]] 

        degree = [[degree_vocab[n] for n in sentences[i][3].split()]]

        dist = [distance_vocab[n] for n in sentences[i][4].split()]
        d = int(math.sqrt(len(dist)))
        MD_r = torch.IntTensor(list(map(int, dist))).view(d, d).tolist()
        if len(MD_r) < src_len:
            num_gap = src_len - len(MD_r)
            list_g1 = [82 for x in range(num_gap)]
            for r in range(len(MD_r)):
                MD_r[r].extend(list_g1)
            list_g2 = [82 for x in range(src_len)]
            for c in range(num_gap):
                MD_r.append(list_g2)

        dist_MA = list(map(int, sentences[i][5].split()))
        d2 = int(math.sqrt(len(dist_MA)))
        MA_r = torch.IntTensor(list(map(int, dist_MA))).view(d2, d2).tolist()
        if len(MA_r) < tgt_len:
            num_gap = tgt_len - len(MA_r)
            list_g1 = [2 for x in range(num_gap)]
            for r in range(len(MA_r)):
                MA_r[r].extend(list_g1)
            list_g2 = [2 for x in range(tgt_len)]
            for c in range(num_gap):
                MA_r.append(list_g2)


        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

        degree_s.extend(degree)
        MD.append(MD_r)
        MA.append(MA_r)

    return torch.IntTensor(enc_inputs), torch.IntTensor(dec_inputs), torch.IntTensor(dec_outputs), torch.IntTensor(degree_s), torch.IntTensor(MD), torch.IntTensor(MA)


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs, degree_s, MD, MA):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

        self.degree_s = degree_s
        self.MD = MD
        self.MA = MA


    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx], self.degree_s[idx], self.MD[idx], self.MA[idx]


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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  
    return pad_attn_mask.expand(batch_size, len_q, len_k)  

def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) 
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask 


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  
        scores.masked_fill_(attn_mask, np.half(-1e9))
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) 
        return context, attn

class ScaledDotProductAttention_encoder(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention_encoder, self).__init__()

    def forward(self, Q, K, V, attn_mask, MD_bias):

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        scores = scores + MD_bias
        scores.masked_fill_(attn_mask, -1e9) 

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2) 
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2) 
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2) 

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  
        output = self.fc(context)  
        return nn.LayerNorm(d_model).cuda(args.local_rank)(output + residual), attn

class MultiHeadAttention_encoder(nn.Module):
    def __init__(self):
        super(MultiHeadAttention_encoder, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask, MD_bias):

        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2) 
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2) 
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2) 

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  

        context, attn = ScaledDotProductAttention_encoder()(Q, K, V, attn_mask, MD_bias)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  
        output = self.fc(context)  
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

        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda(args.local_rank)(output + residual) 


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention_encoder()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask, MD_bias):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask, MD_bias) 
        enc_outputs = self.pos_ffn(enc_outputs) 
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  
        return dec_outputs, dec_self_attn, dec_enc_attn


# New
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.degree_s = nn.Embedding(degree_vocab_size, d_model, padding_idx=7)
        self.MD = nn.Embedding(distance_vocab_size, n_heads, padding_idx=82)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs, degree_s, MD):

        enc_outputs = self.src_emb(enc_inputs) 
        enc_outputs = enc_outputs + self.degree_s(degree_s)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) 
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) 
        enc_self_attns = []
        MD_bias = self.MD(MD).permute(0, 3, 1, 2) 
        for layer in self.layers:

            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask, MD_bias)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):

        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda(args.local_rank)  
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda(args.local_rank) 
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda(args.local_rank)  
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).cuda(args.local_rank)  

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = Linear(d_model*2, 512)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        self.hidden2 = Linear(512, 32)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.hidden3 = Linear(32, 2)
        xavier_uniform_(self.hidden3.weight)


    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        return X


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda(args.local_rank)
        self.decoder = Decoder().cuda(args.local_rank)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda(args.local_rank)
        self.mlp = MLP().cuda(args.local_rank)
        self.BCE = nn.BCEWithLogitsLoss().cuda(args.local_rank)

    def forward(self, enc_inputs, dec_inputs, degree_s, MD, MA):
 
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, degree_s, MD)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        c = []
        t = []
        for n in range(dec_outputs.shape[0]):
            num_t = MA[n][0].tolist().index(2)
            for i in range(num_t):
                for j in range(i+1, num_t):
                    a1 = dec_outputs[n][i+1]
                    a2 = dec_outputs[n][j+1]
                    b = torch.cat((a1, a2), dim=0)
                    c.append(b.unsqueeze(0))
            for i in range(num_t-1):
                t.extend(MA[n][i][i+1:num_t])
        MLP_input = torch.cat(c, dim=0)
        MLP_output = self.mlp(MLP_input)
        target = torch.LongTensor([0 if i == -1 else i for i in t])
        onehot_target = torch.eye(2)[target.long(), :].cuda(args.local_rank)
        sum_loss = self.BCE(MLP_output, onehot_target)

        dec_logits = self.projection(dec_outputs) 
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns, MLP_output, sum_loss


def greedy_decoder(model, enc_input, start_symbol):

    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = True
    next_symbol = start_symbol
    t_n = 0
    max_len  = 120 
    while terminal == True and max_len > 0:
        max_len = max_len - 1
        t_n = t_n + 1
        dec_input = torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype)].cuda(args.local_rank),-1)
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
        if torch.cuda.device_count() >= 1:
            logging.warning((f"Find {torch.cuda.device_count()} GPUs!"))
        else:
            logging.warning("Too few GPUs!")
            sys.exit(0)
    else:
        logging.warning("Cuda is not available! Exit!")
        sys.exit(0)

    n_gpus = 1
    torch.distributed.init_process_group("nccl", world_size=n_gpus, rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)
    model = Transformer()
    model = nn.parallel.DistributedDataParallel(model.cuda(args.local_rank),
                                                device_ids=[args.local_rank])  # DistributedDataParallel

    # 读取train数据
    data_t = pd.read_csv(r"/home/user06/model/data/pre-train/enc_dec_in_dec_out.csv")
    data_d = pd.read_csv(r"/home/user06/model/data/pre-train/degree_enc.csv")
    data_MD = pd.read_csv(r"/home/user06/model/data/pre-train/MD.csv")
    data_MA = pd.read_csv(r"/home/user06/model/data/pre-train/MA.csv")
    Sentences = []
    for i in range(len(data_t.index)): 
        sentence = []
        sentence.append(data_t["enc_inputs"][i])
        sentence.append(data_t["dec_inputs"][i])
        sentence.append(data_t["dec_outputs"][i])

        # New
        sentence.append(data_d["degree"][i])
        sentence.append(data_MD["MD"][i])
        sentence.append(data_MA["MA"][i])
        Sentences.append(sentence)
    random.shuffle(Sentences)
    enc_inputs, dec_inputs, dec_outputs, degree_s, MD, MA = make_data(Sentences)
    train_sampler = DistributedSampler(MyDataSet(enc_inputs, dec_inputs, dec_outputs, degree_s, MD, MA))
    batch_size = 180 
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs, degree_s, MD, MA), batch_size=batch_size,
                             sampler=train_sampler)


    data_v = pd.read_csv(r"/home/user06/model/data/valid data_all.csv")
    Sentences_v = []
    for i in range(len(data_v.index)):
        sentence_v = []
        sentence_v.append(data_v["enc_inputs"][i])
        sentence_v.append(data_v["dec_inputs"][i])
        sentence_v.append(data_v["dec_outputs"][i])
        sentence_v.append(data_v["degree"][i])
        sentence_v.append(data_v["MD"][i])
        sentence_v.append(data_v["MA"][i])
        Sentences_v.append(sentence_v)
    enc_inputs_v, dec_inputs_v, dec_outputs_v, degree_v, MD_v, MA_v = make_data(Sentences_v)
    val_sampler = DistributedSampler(MyDataSet(enc_inputs_v, dec_inputs_v, dec_outputs_v, degree_v, MD_v, MA_v))
    batch_size_v = 1
    loader_v = Data.DataLoader(MyDataSet(enc_inputs_v, dec_inputs_v, dec_outputs_v, degree_v, MD_v, MA_v), batch_size=batch_size_v,
                               sampler=val_sampler)


    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.01, betas=(0.9, 0.99), eps=1e-06, weight_decay=0,
                                  foreach=None)
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda step: rate(step, d_model, factor=100,
                                                                                 warmup=2000),
                                                     last_epoch=-1)
    start_time = time.time()
    scaler = GradScaler()
    
    step = 0
    for epoch in range(100):
        print("----------{} start ----------".format(epoch+1))
        train_sampler.set_epoch(epoch)  
        sum_loss = 0
        for enc_inputs, dec_inputs, dec_outputs, degree_s, MD, MA in loader:

          optimizer.zero_grad()

          step = step + 1
          enc_inputs, dec_inputs, dec_outputs, degree_s, MD, MA = enc_inputs.cuda(args.local_rank), dec_inputs.cuda(
              args.local_rank), dec_outputs.cuda(args.local_rank), degree_s.cuda(args.local_rank), MD.cuda(args.local_rank), MA.cuda(args.local_rank)
          with autocast(enabled=True):
              outputs, enc_self_attns, dec_self_attns, dec_enc_attns, MLP_output, loss_MLP = model(enc_inputs, dec_inputs, degree_s, MD, MA)
          loss1 = criterion(outputs, dec_outputs.view(-1).long())
          loss = loss1 + loss_MLP
          writer.add_scalar('loss', loss.item(), global_step=step+1)
          scaler.scale(loss).backward()
          scaler.unscale_(optimizer)
          scaler.step(optimizer)
          scaler.update()
          lr_scheduler.step()

          sum_loss = sum_loss + loss.item()
          if step % 1 == 0:
              print('Epoch:', '%04d' % (epoch + 1), 'step:', '%04d' % (step), 'loss =', '{:.6f}'.format(loss.item()))
              writer.add_scalar('lr_scheduler', optimizer.state_dict()["param_groups"][0]["lr"], global_step=step + 1)
              print('learning rate:', optimizer.state_dict()["param_groups"][0]["lr"])
          
          if step % 5 == 0:
                loss_v_sum_1 = 0
                loss_v_sum_2 = 0
                for enc_inputs_v, dec_inputs_v, dec_outputs_v, degree_v, MD_v, MA_v in loader_v:
                    enc_inputs_v, dec_inputs_v, dec_outputs_v, degree_v, MD_v, MA_v = enc_inputs_v.cuda(args.local_rank), dec_inputs_v.cuda(
                        args.local_rank), dec_outputs_v.cuda(args.local_rank), degree_v.cuda(args.local_rank), MD_v.cuda(
                        args.local_rank), MA_v.cuda(args.local_rank)
                    with torch.no_grad():
                        outputs_v, enc_self_attns_v, dec_self_attns_v, dec_enc_attns_v, MLP_output_v, loss_MLP_v = model(enc_inputs_v, dec_inputs_v, degree_v, MD_v, MA_v)
                    loss_v = criterion(outputs_v, dec_outputs_v.view(-1).long())
                    loss_v_sum_1 = loss_v_sum_1 + loss_v
                    loss_v_sum_2 = loss_v_sum_2 + loss_MLP_v
                loss_avg_v_1 = loss_v_sum_1 / 210
                loss_avg_v_2 = loss_v_sum_2 / 210
                writer.add_scalar('loss_val_1', loss_avg_v_1, global_step=step + 1)
                writer.add_scalar('loss_val_2', loss_avg_v_2, global_step=step + 1)
                print("loss_val_1:{}".format(loss_avg_v_1))
                print("loss_val_2:{}".format(loss_avg_v_2))
          
          if (step) % 1 == 0:
              print("a_step_speed:", time.time() - start_time)
              start_time = time.time()
          if (step) % 100 == 0 and args.local_rank == 0:
            print('step:', step)
            print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
            print('learning rate:', lr_scheduler.get_last_lr()[0])
            checkpoint = {
                "model": model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                "step": step,
                'lr_schedule': lr_scheduler.state_dict()
            }
            save_path = './model'
            filename = 'ckpt_step_{}.pth'.format(step)
            torch.save(checkpoint, os.path.join(save_path, filename))
            print("model save")