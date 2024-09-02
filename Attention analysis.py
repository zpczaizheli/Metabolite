import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import time
import os
import pandas as pd
from torch.nn import Linear, ReLU, Sigmoid
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from rdkit import rdBase, Chem, DataStructs
from rdkit import rdBase, Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
import copy
import itertools

from matplotlib import pyplot as plt
import seaborn as sns


Words1 = ['Pad', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se',
          'Zn', 'H', 'Cu',
          'Mn', '-', '=', '#', '~', '?', '?5', '.', '>', 'Pd', 'Li', 'Sn', 'Mo', 'Co', 'Cs', 'As']
Words2 = ['Pad', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se',
          'Zn', 'H', 'Cu',
          'Mn', '-', '=', '#', '~', '?', '?5', '.', 'sos', 'eos', 'Pd', 'Li', 'Sn', 'Mo', 'Co', 'Cs', 'As']

src_vocab = {'Pad': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Si': 6, 'P': 7, 'Cl': 8, 'Br': 9, 'Mg': 10,
             'Na': 11, 'Ca': 12, 'Fe': 13, 'Al': 14, 'I': 15, 'B': 16, 'K': 17, 'Se': 18, 'Zn': 19, 'H': 20,
             'Cu': 21, 'Mn': 22, '-': 23, '=': 24, '#': 25, '~': 26, 'Pd': 27, 'Li': 28, 'Sn': 29, 'Mo': 30, 'Co': 31,
             'Cs': 32, 'As': 33}
src_vocab_size = len(src_vocab)
src_idx2word = {i: w for i, w in enumerate(src_vocab)}

tgt_vocab = {'Pad': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Si': 6, 'P': 7, 'Cl': 8, 'Br': 9, 'Mg': 10,
             'Na': 11, 'Ca': 12, 'Fe': 13, 'Al': 14, 'I': 15, 'B': 16, 'K': 17, 'Se': 18, 'Zn': 19, 'H': 20,
             'Cu': 21, 'Mn': 22, '-': 23, '=': 24, '#': 25, '~': 26, 'sos': 27, 'eos': 28, 'Pd': 29, 'Li': 30, 'Sn': 31,
             'Mo': 32, 'Co': 33, 'Cs': 34, 'As': 35}
tgt_vocab_size = len(tgt_vocab)
idx2word = {i: w for i, w in enumerate(tgt_vocab)}

# degree coding New
degree_vocab = {'pad': 7, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}
degree_vocab_size = len(degree_vocab)

# distance coding New
distance_vocab = {'-2': 82, '-1': 81, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                  '10': 10,
                  '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20,
                  '21': 21, '22': 22, '23': 23, '24': 24, '25': 25, '26': 26, '27': 27, '28': 28, '29': 29, '30': 30,
                  '31': 31, '32': 32, '33': 33, '34': 34, '35': 35, '36': 36, '37': 37, '38': 38, '39': 39, '40': 40,
                  '41': 41, '42': 42, '43': 43, '44': 44, '45': 45, '46': 46, '47': 47, '48': 48, '49': 49, '50': 50,
                  '51': 51, '52': 52, '53': 53, '54': 54, '55': 55, '56': 56, '57': 57, '58': 58, '59': 59, '60': 60,
                  '61': 61, '62': 62, '63': 63, '64': 64, '65': 65, '66': 66, '67': 67, '68': 68, '69': 69, '70': 70,
                  '71': 71, '72': 72, '73': 73, '74': 74, '75': 75, '76': 76, '77': 77, '78': 78, '79': 79, '80': 80}
distance_vocab_size = len(distance_vocab)

src_len = 132 
tgt_len = 107 


d_model = 512  
d_ff = 2048  
d_k = d_v = 64 
n_layers = 6 
n_heads = 8 



def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs, degree_s, MD, MA = [], [], [], [], [], []
    for i in range(len(sentences)):
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


        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

        degree_s.extend(degree)
        MD.append(MD_r)
        MA.append(1)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs), torch.LongTensor(
        degree_s), torch.LongTensor(MD), torch.LongTensor(MA)


# New
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
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx], self.degree_s[idx], self.MD[idx], \
               self.MA[idx]


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
        scores.masked_fill_(attn_mask, -1e9)

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
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)  

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1) 

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  
        output = self.fc(context) 
        return nn.LayerNorm(d_model)(output + residual), attn


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
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2) 

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1) 

        context, attn = ScaledDotProductAttention_encoder()(Q, K, V, attn_mask, MD_bias)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v) 
        output = self.fc(context)  
        return nn.LayerNorm(d_model)(output + residual), attn


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
        return nn.LayerNorm(d_model)(output + residual) 


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
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) 
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)  
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0)  

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
       
        self.hidden1 = Linear(d_model * 2, 128)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        
        self.hidden2 = Linear(128, 16)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        
        self.hidden3 = Linear(16, 2)
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
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.mlp = MLP()
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, enc_inputs, dec_inputs, degree_s, MD, MA):

        enc_outputs, enc_self_attns = self.encoder(enc_inputs, degree_s, MD)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        
        c = []
        t = []
        for n in range(dec_outputs.shape[0]):
            num_t = MA[n][0].tolist().index(2)
            for i in range(num_t):
                for j in range(i + 1, num_t):
                    a1 = dec_outputs[n][i + 1]
                    a2 = dec_outputs[n][j + 1]
                    b = torch.cat((a1, a2), dim=0)
                    c.append(b.unsqueeze(0))
            for i in range(num_t - 1):
                t.extend(MA[n][i][i + 1:num_t])
        MLP_input = torch.cat(c, dim=0)
        MLP_output = self.mlp(MLP_input)
        target = torch.LongTensor([0 if i == -1 else i for i in t])
        onehot_target = torch.eye(2)[target.long(), :]
        sum_loss = self.BCE(MLP_output, onehot_target)

        dec_logits = self.projection(dec_outputs) 
        return dec_logits.view(-1,
                               dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns, MLP_output, sum_loss



def generate_SA(smi):
    Mol = Chem.MolFromSmiles(smi)
    AtomsNum = Mol.GetNumAtoms()

    ab_list = []
    B_A = []

    for index1 in range(AtomsNum):
        num_input_atom = 0
        Atom = Mol.GetAtomWithIdx(index1)
        AtomSymble = Atom.GetSymbol()
        for index2 in range(index1 - 1, -1, -1):
            Bond = Mol.GetBondBetweenAtoms(index1, index2)
            if str(Bond) == "None":
                continue
            else:
                B_A.append([index1, index2])
                BondType = Bond.GetBondType()
                if str(BondType) == 'SINGLE':
                    ab_list.append("-")
                elif str(BondType) == 'DOUBLE':
                    ab_list.append("=")
                elif str(BondType) == 'TRIPLE':
                    ab_list.append("#")
                elif str(BondType) == 'AROMATIC':
                    ab_list.append("~")
                if num_input_atom == 0:
                    ab_list.append(AtomSymble)
                    num_input_atom = num_input_atom + 1

        if num_input_atom == 0:
            ab_list.append(AtomSymble)
            num_input_atom = num_input_atom + 1

    num_v = len(ab_list)

    bond_list = ["=", '-', '~', "#"]
    A_list = []
    B_list = []
    for i in range(len(ab_list)):
        if ab_list[i] not in bond_list:
            A_list.append(i)
        else:
            B_list.append(i)

    mol = Chem.MolFromSmiles(smi)
    AM2 = Chem.GetAdjacencyMatrix(mol)
    AM2 = torch.from_numpy(AM2)


    MA = (torch.ones(num_v, num_v, dtype=torch.int)) * -1

    for i in range(num_v):
        MA[i][i] = 0

    ab_list2 = []
    for index1 in range(AtomsNum):
        num_input_atom = 0
        Atom = Mol.GetAtomWithIdx(index1)
        AtomSymble = Atom.GetSymbol()
        for index2 in range(index1 - 1, -1, -1):
            Bond = Mol.GetBondBetweenAtoms(index1, index2)
            if str(Bond) == "None":
                continue
            else:
                MA[len(ab_list2)][A_list[index1]] = 1
                MA[len(ab_list2)][A_list[index2]] = 1
                MA[A_list[index1]][len(ab_list2)] = 1
                MA[A_list[index2]][len(ab_list2)] = 1
                BondType = Bond.GetBondType()
                if str(BondType) == 'SINGLE':
                    ab_list2.append("-")
                elif str(BondType) == 'DOUBLE':
                    ab_list2.append("=")
                elif str(BondType) == 'TRIPLE':
                    ab_list2.append("#")
                elif str(BondType) == 'AROMATIC':
                    ab_list2.append("~")
                if num_input_atom == 0:
                    ab_list2.append(AtomSymble)
                    num_input_atom = num_input_atom + 1

        if num_input_atom == 0:
            ab_list2.append(AtomSymble)
            num_input_atom = num_input_atom + 1
    return ab_list, MA


def SymbolAtom(mol):
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp('molAtomMapNumber', str(i))
    return mol


def SymbolAtom2(mol):
    Atomnum = mol.GetNumAtoms()
    for i in range(Atomnum):
        Atom = mol.GetAtomWithIdx(i)
        Atom.SetProp("atomNote", str(i))
    return mol


def seq_pain(A, ab):
    Symbol_ID_Dic = {}
    Symbol_ID_Dic['C'] = 6
    Symbol_ID_Dic['N'] = 7
    Symbol_ID_Dic['O'] = 8
    Symbol_ID_Dic['S'] = 16
    Symbol_ID_Dic['F'] = 9
    Symbol_ID_Dic['Si'] = 14
    Symbol_ID_Dic['P'] = 15
    Symbol_ID_Dic['Cl'] = 17
    Symbol_ID_Dic['Br'] = 35
    Symbol_ID_Dic['Mg'] = 12
    Symbol_ID_Dic['Na'] = 11
    Symbol_ID_Dic['Ca'] = 20
    Symbol_ID_Dic['Fe'] = 26
    Symbol_ID_Dic['AL'] = 13
    Symbol_ID_Dic['I'] = 53
    Symbol_ID_Dic['B'] = 5
    Symbol_ID_Dic['K'] = 19
    Symbol_ID_Dic['Se'] = 34
    Symbol_ID_Dic['Zn'] = 30
    Symbol_ID_Dic['H'] = 1
    Symbol_ID_Dic['Cu'] = 29
    Symbol_ID_Dic['Mn'] = 25
    Symbol_ID_Dic['Pd'] = 46
    Symbol_ID_Dic['Li'] = 3
    Symbol_ID_Dic['Sn'] = 50
    Symbol_ID_Dic['Mo'] = 42
    Symbol_ID_Dic['Co'] = 27
    Symbol_ID_Dic['Cs'] = 55
    Symbol_ID_Dic['As'] = 33

    Mol = Chem.MolFromSmiles(ab[0])
    mw = Chem.RWMol(Mol)
    bond = ["-", "=", "#", "~"]
    for i in range(1, len(ab)):
        if ab[i] not in bond:
            mw.AddAtom(Chem.Atom(Symbol_ID_Dic[ab[i]]))

    a_dic = {}
    n_a = 0
    for i in range(len(ab)):
        if ab[i] not in bond:
            a_dic[i] = n_a
            n_a = n_a + 1
    for i in range(len(ab)):
        if ab[i] in bond:
            a2 = []
            for j in range(len(ab)):
                if A[i][j] == 1:
                    a2.append(j)
            if ab[i] == '-':
                mw.AddBond(a_dic[a2[0]], a_dic[a2[1]], Chem.BondType.SINGLE)
            elif ab[i] == '=':
                mw.AddBond(a_dic[a2[0]], a_dic[a2[1]], Chem.BondType.DOUBLE)
            elif ab[i] == '#':
                mw.AddBond(a_dic[a2[0]], a_dic[a2[1]], Chem.BondType.TRIPLE)
            elif ab[i] == '~':
                mw.AddBond(a_dic[a2[0]], a_dic[a2[1]], Chem.BondType.AROMATIC)

    m_edit = mw.GetMol()
    mol_ = SymbolAtom2(m_edit) 
    s = Chem.MolToSmiles(mol_)
    return s


data_t = pd.read_csv("/home/user06/model/analyze/example input.csv")

Sentences = []
for i in range(len(data_t.index)):  # len(data_t.index)
    sentence = []
    sentence.append(data_t["Drug"][i])
    sentence.append("C")
    sentence.append("C")
    sentence.append(data_t["Degree"][i])
    sentence.append(data_t["MD"][i])
    Sentences.append(sentence) 

enc_inputs, dec_inputs, dec_outputs, degree_s, MD, MA = make_data(Sentences)
loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs, degree_s, MD, MA), 100, False)
enc_inputs, _, _, degree, MD, _ = next(iter(loader))


class Beamheap:
    def fun(que, x, k=40):  
        que.append(x)

        def second(k):  
            return k[1]

        que.sort(key=second)
        if len(que) > k:
            que = que[-k:]
        return que

 

def attention_plot(attention, x_texts, y_texts=None, figsize=(15, 10), annot=False, figure_path='./figures',
                   figure_name='attention_weight.png'):
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(attention,
                     cbar=True,
                     cmap="RdBu_r",
                     annot=annot,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 10},
                     yticklabels=y_texts,
                     xticklabels=x_texts
                     )
    hm.xaxis.tick_top()
    if os.path.exists(figure_path) is False:
        os.makedirs(figure_path)
    plt.savefig(os.path.join(figure_path, figure_name))
    plt.close()


class beam_search_decoder:
    def __init__(self):
        self.topk = Beamheap.fun  

    def forward(self, x0, enc_inputs, degree, MD, max_len=150):  

        enc_outputs, enc_self_attns = model.encoder(enc_inputs, degree, MD)

        vector = np.zeros([1, 1, 512])
        vector = torch.LongTensor(vector)
        beams = [([x0], 0.0, vector, vector, vector)]  
        for _ in range(max_len):  
            que = [] 
            for x, score, v, att, self_attn in beams:  
                if x[-1] == 28:  
                    que = self.topk(que, (x, score, v, att, self_attn))
                else:
                    y = torch.from_numpy(np.array(x))
                    dec_input = y.unsqueeze(0)
                    dec_outputs, dec_self_attns, dec_enc_attns = model.decoder(dec_input, enc_inputs,
                                                                               enc_outputs)  
                    v = torch.cat([v, dec_outputs[:, -1:, :]], 1)
                    projected_r = model.projection(dec_outputs)
                    projected = projected_r[-1, -1, :]
                    projected = F.log_softmax(projected)
                    output = projected.tolist()
                    for i, o_score in enumerate(output): 

                        que = self.topk(que, (
                            x + [i], score + o_score, v, dec_enc_attns,
                            dec_self_attns)) 
            beams = que  

        return beams, enc_self_attns


def lizi(smi):
    t = smi.count("n")
    t2 = smi.count("N")
    paths = list(itertools.product(["n", "[n-]"], repeat=t))
    paths2 = list(itertools.product(["N", "[N-]"], repeat=t2))
    meta_n = []
    meta_N = []
    for path in paths:
        num = 0
        seq = copy.deepcopy(smi)
        seq = list(seq)
        for i in range(len(smi)):
            if seq[i] == "n":
                seq[i] = path[num]
                num = num + 1
        seq = ''.join(seq)
        meta_n.append(seq)
    for path in paths2:
        for i in meta_n:
            seq = list(i)
            num = 0
            for j in range(len(seq)):
                if seq[j] == "N":
                    seq[j] = path[num]
                    num = num + 1
            seq = ''.join(seq)
            meta_N.append(seq)
    for s in meta_n:
        try:
            mol = Chem.MolFromSmiles(s)
            mol = SymbolAtom(mol)
            return s
        except AttributeError:
            continue
    for s in meta_N:
        try:
            mol = Chem.MolFromSmiles(s)
            mol = SymbolAtom(mol)
            return s
        except AttributeError:
            continue
    paths3 = list(itertools.product(["n", "[n-]", "[n+]"], repeat=t))
    paths4 = list(itertools.product(["N", "[N-]", "[N+]"], repeat=t2))
    meta_n2 = []
    meta_N2 = []
    for path in paths3:
        num = 0
        seq = copy.deepcopy(smi)
        seq = list(seq)
        for i in range(len(smi)):
            if seq[i] == "n":
                seq[i] = path[num]
                num = num + 1
        seq = ''.join(seq)
        meta_n2.append(seq)
    for path in paths4:
        for i in meta_n:
            seq = list(i)
            num = 0
            for j in range(len(seq)):
                if seq[j] == "N":
                    seq[j] = path[num]
                    num = num + 1
            seq = ''.join(seq)
            meta_N2.append(seq)
    for s in meta_n2:
        try:
            mol = Chem.MolFromSmiles(s)
            mol = SymbolAtom(mol)
            return s
        except AttributeError:
            continue
    for s in meta_N2:
        try:
            mol = Chem.MolFromSmiles(s)
            mol = SymbolAtom(mol)
            return s
        except AttributeError:
            continue
    return smi



def getAccLineChart(x, y, save_dir, chart_name):
    l = plt.plot(x, y, label="")
    plt.xlabel("Node")
    plt.ylabel("Attn_score")
    plt.xticks(rotation=90, size=5) 
    plt.legend()
    plt.grid(True)
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, chart_name), dpi=500)

def divide_rows_by_diagonal(matrix):
    diagonal = torch.diag(matrix) 
    result = matrix / diagonal.view(-1, 1)  
    return result

if __name__ == "__main__":
    metas = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']
    data_m = pd.read_csv("/home/user06/model/analyze/Molecular metabolites are tested.csv")

    sigmoid = nn.Sigmoid()

    smi_sum_hit_n = 0
    smi_hit_one_n = 0
    smi_hit_half_n = 0
    smi_hit_all_n = 0

    each_hit = []
    smi_sum_hit = 0
    smi_hit_one = 0
    smi_hit_half = 0
    smi_hit_all = 0

    sum_valid = 0
    sum_invaild = 0
    con_list = [21, 22, 31, 34, 40, 42, 52, 54, 55, 61]
    hyd_list = [32, 36, 39, 40, 51, 52, 56, 63]
    model_list = ["/home/user06/model/oxidation/8665_ckpt_step_3900.pth", "/home/user06/model/combination/原_ckpt_step_1800.pth",
                  "/home/user06/model/hydrolysis/ckpt_step_800.pth"]
    for i in range(len(enc_inputs)): 
        print("第{}个".format(i + 1))
        print("Source:", data_m["Drug"][i])
        num = 0
        m = []
        m_smiles = []  
        for j in metas:
            meta = data_m[j][i]
            if meta != "N":
                m_smiles.append(meta)
                num = num + 1
                m.append(meta)
                print("output_t{}:{}".format(num, meta))

        smi_hit_list_n = []
        smi_hit_order_n = []

        top15_list = []
        smi_hit_list = []
        smi_hit_order = []

        drug_text, _ = generate_SA(data_m["Drug"][i])

        for mod in range(len(model_list)):
            ckpt = torch.load(model_list[mod], map_location='cpu')
            model = Transformer()
            model.load_state_dict(ckpt["model"])
            beam = beam_search_decoder()
            x1 = enc_inputs[i].view(1, -1)
            x2 = degree[i].view(1, -1)
            x3 = MD[i].unsqueeze(0)
            b, enc_self_attentions = beam.forward(27, x1, x2, x3)
            enc_self_attentions = enc_self_attentions[5]

            enc_self_attentions = enc_self_attentions[:, :, :len(drug_text), :len(drug_text)]

            for j in range(len(b)):
                predict_r = [idx2word[n] for n in b[len(b) - j - 1][0]]
                predict_r_list = predict_r[1:-1]

                vector = b[len(b) - j - 1][2]
                num_v = vector.shape[1] - 2
                bond_type = ["-", "=", "#", "~"]
                MA_pred = np.ones([num_v, num_v], dtype=np.int32) * -1
                for x in range(num_v):
                    MA_pred[x][x] = 0
                MA_score_0 = np.zeros([num_v, num_v])
                MA_score_1 = np.zeros([num_v, num_v])
                num_bond_list = [0 for x in range(num_v)]
                for x in range(num_v):
                    for y in range(x + 1, num_v):
                        if (predict_r_list[x] in bond_type and predict_r_list[y] not in bond_type) or (
                                predict_r_list[x] not in bond_type and predict_r_list[y] in bond_type):
                            # if predict_r_list[x] == "C" and num_bond_list[x] >= 4:
                            #     continue
                            a1 = vector[0][x + 2]
                            a2 = vector[0][y + 2]
                            mlp_input = torch.cat((a1, a2), dim=0)
                            mlp_output = model.mlp(mlp_input)
                            sigmoid_pred = sigmoid(mlp_output)
                            MA_score_0[x][y] = sigmoid_pred[0]
                            MA_score_1[x][y] = sigmoid_pred[1]
                            if sigmoid_pred[1] > sigmoid_pred[0]:
                                MA_pred[x][y] = 1
                                MA_pred[y][x] = 1
                                if predict_r_list[y] == "-":
                                    num_bond_list[x] = num_bond_list[x] + 1
                                    num_bond_list[y] = num_bond_list[y] + 1 
                                elif predict_r_list[y] == "=":
                                    num_bond_list[x] = num_bond_list[x] + 2
                                    num_bond_list[y] = num_bond_list[y] + 1
                                elif predict_r_list[y] == "#":
                                    num_bond_list[x] = num_bond_list[x] + 3
                                    num_bond_list[y] = num_bond_list[y] + 1
                                elif predict_r_list[y] == "~":
                                    num_bond_list[x] = num_bond_list[x] + 1.5
                                    num_bond_list[y] = num_bond_list[y] + 1
                                else:
                                    num_bond_list[x] = num_bond_list[x] + 1
                                    if predict_r_list[x] == "-":
                                        num_bond_list[y] = num_bond_list[y] + 1
                                    elif predict_r_list[x] == "=":
                                        num_bond_list[y] = num_bond_list[y] + 2
                                    elif predict_r_list[x] == "#":
                                        num_bond_list[y] = num_bond_list[y] + 3
                                    elif predict_r_list[x] == "~":
                                        num_bond_list[y] = num_bond_list[y] + 1.5
               
                for x in range(num_v):
                    if predict_r_list[x] in bond_type and num_bond_list[x] == 1:
                        max_v = []
                        for y in range(x, -1, -1):
                            if MA_pred[y][x] != 1:
                                max_v.append((MA_score_1[y][x], y, x))
                        max_v.sort(key=lambda z: z[0], reverse=True)
                        MA_pred[max_v[0][1]][max_v[0][2]] = 1
                        MA_pred[max_v[0][2]][max_v[0][1]] = 1
                        if len(max_v) >= 2: 
                            if max_v[1][0] > 0.17 and max_v[1][1] > max_v[0][1]:
                                MA_pred[max_v[0][1]][max_v[0][2]] = -1
                                MA_pred[max_v[0][2]][max_v[0][1]] = -1
                                MA_pred[max_v[1][1]][max_v[1][2]] = 1
                                MA_pred[max_v[1][2]][max_v[1][1]] = 1
                    if predict_r_list[x] == "C" and num_bond_list[x] == 0:
                        max_v2 = []
                        for y in range(x, -1, -1):
                            if MA_pred[y][x] != 1:
                                max_v2.append((MA_score_1[y][x], y, x))
                        max_v2.sort(key=lambda z: z[0], reverse=True)
                        MA_pred[max_v2[0][1]][max_v2[0][2]] = 1
                        MA_pred[max_v2[0][2]][max_v2[0][1]] = 1
                        num_bond_list[x] = num_bond_list[x] + 1

                    if predict_r_list[x] == "-" or predict_r_list[x] == '~':
                        if num_bond_list[x] == 3:
                            max_v3 = []
                            for y in range(x, -1, -1):
                                if MA_pred[y][x] == 1:
                                    max_v3.append((y, x))
                            if len(max_v3) == 2:
                                MA_pred[max_v3[1][0]][max_v3[1][1]] = -1
                                MA_pred[max_v3[1][1]][max_v3[1][0]] = -1
                            if len(max_v3) == 3:
                                MA_pred[max_v3[2][0]][max_v3[2][1]] = -1
                                MA_pred[max_v3[2][1]][max_v3[2][0]] = -1

                try:
                    smi_pred_1 = seq_pain(MA_pred, predict_r_list)
                    smi_pred = lizi(smi_pred_1)
                    p_smi = Chem.MolFromSmiles(smi_pred)
                    mol = SymbolAtom(p_smi)

                    s_mol = Chem.MolFromSmiles(data_m["Drug"][i])
                    t_mol = Chem.MolFromSmiles(smi_pred)
                    num_s = s_mol.GetNumAtoms()
                    num_t = t_mol.GetNumAtoms()

                    ms = [s_mol, t_mol]
                    fps = [Chem.RDKFingerprint(x) for x in ms]
                    gb = DataStructs.FingerprintSimilarity(fps[0], fps[1])

                   
                    if mod == 0:
                        if num_t / num_s > 0.5 and 0.494 < gb < 1: 
                            top15_list.append((b[len(b) - j - 1][1], smi_pred, "oxidation", predict_r, b[len(b) - j - 1][3],
                                               b[len(b) - j - 1][4]))
                    elif mod == 1 and num_t - num_s >= 3:
                        if num_t / num_s > 1.1 and 0.2 < gb < 1: 
                            top15_list.append((b[len(b) - j - 1][1], smi_pred, "combination", predict_r, b[len(b) - j - 1][3],
                                               b[len(b) - j - 1][4]))
                    elif mod == 2 and num_s - num_t >= 2:
                        if num_t / num_s > 0.33 and 0.52 < gb < 1:  
                            top15_list.append((b[len(b) - j - 1][1], smi_pred, "hydrolysis", predict_r, b[len(b) - j - 1][3],
                                               b[len(b) - j - 1][4]))
                except (OverflowError, KeyError, AttributeError, IndexError, RuntimeError, ValueError) as e:
                    sum_invaild = sum_invaild + 1

        top15_list.sort(key=lambda z: z[0], reverse=True)

        top_all = []
        top_all_smi = []
        for element in top15_list:
            if element[2] == "oxidation":
                top_all_smi.append(element[1])
                top_all.append(element)
            elif element[2] == "combination" and element[0] > -2:  
                top_all_smi.append(element[1])
                top_all.append(element)
            elif element[2] == "hydrolysis" and element[0] > -2: 
                top_all_smi.append(element[1])
                top_all.append(element)

        print("top15_list_len:", len(top15_list))
        print("top_de_r:", len(top_all))
        sum_valid = sum_valid + len(top_all)

        drug_text, _ = generate_SA(data_m["Drug"][i])
        for k in range(len(m_smiles)):
            print(k)
            t_smiles = Chem.MolFromSmiles(m_smiles[k])
            for p in range(len(top_all)):  
                meta_text = top_all[p][3]
                attentions = top_all[p][4][5][:, :, :len(meta_text), :len(drug_text)]
                max_attentions = F.normalize(attentions[-1, 0, :, :], p=2, dim=-1)
                for ii in range(attentions.shape[1]):
                    attentions_norm = F.normalize(attentions[-1, ii, :, :], p=2, dim=-1)
                    if ii > 0:
                        for jj in range(attentions.shape[2]):
                            for kk in range(attentions.shape[3]):
                                if max_attentions[jj][kk] < attentions_norm[jj][kk]:
                                    max_attentions[jj][kk] = attentions_norm[jj][kk]
                attention_plot(max_attentions.detach().numpy(), x_texts=drug_text, y_texts=meta_text,
                               annot=False,
                               figsize=(15, 15), figure_path='./figures/drug{}/predict{}'.format(i, p + 1),
                               figure_name='max_attention.png')
   
                p_smiles = Chem.MolFromSmiles(top_all[p][1])
                if DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(t_smiles),
                                                     Chem.RDKFingerprint(p_smiles)) == 1:
                    smi_hit_list_n.append(k + 1)
                    smi_hit_order_n.append(p + 1)

        for k in range(len(top_all)):
            print("p{}:{}".format(k + 1, top_all[k][:4]))
        print("smi_hit_list_all:", smi_hit_list_n)
        print("smi_hit_order_all:", smi_hit_order_n)

        smi_sum_hit_n = smi_sum_hit_n + len(set(smi_hit_list_n))
        if len(set(smi_hit_list_n)) > 0:
            smi_hit_one_n = smi_hit_one_n + 1
        if len(set(smi_hit_list_n)) >= num / 2:
            smi_hit_half_n = smi_hit_half_n + 1
        if len(set(smi_hit_list_n)) == num:
            smi_hit_all_n = smi_hit_all_n + 1

    print("smi_sum_hit:", smi_sum_hit_n)
    print("smi_hit_one:", smi_hit_one_n)
    print("smi_hit_half:", smi_hit_half_n)
    print("smi_hit_all:", smi_hit_all_n)
    print("sum_vaild:", sum_valid)
    print("sum_invaild:", sum_invaild)

