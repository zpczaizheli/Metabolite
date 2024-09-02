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
from rdkit import rdBase, Chem, DataStructs
import copy
writer = SummaryWriter('runs/exp')
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from torch.nn import Linear, ReLU, Sigmoid
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch.nn.functional as F
import itertools



smi_vocab = {'C': 1, '(': 2, ')': 3, '[Mg+]': 4, '.': 5, 'O': 6, 'c': 7, '1': 8, 'n': 9, '=': 10, 'N': 11, '>': 12, '[Cl-]': 13, '[N+]': 14,
              '[O-]': 15, 'Cl': 16, '2': 17, '-': 18, '3': 19, 'F': 20, 's': 21, '#': 22, 'S': 23, '[nH]': 24, '4': 25, 'Br': 26, '[P-]': 27,
              '[BH4-]': 28, '[Na+]': 29, 'I': 30, '[S-]': 31, '[K+]': 32, '[Al]': 33, '[Li]': 34, 'B': 35, '[H-]': 36, '[OH-]': 37, '[Zn]': 38,
              '[K]': 39, '[Al+]': 40, 'o': 41, '[Cl+]': 42, '[Na]': 43, '[I-]': 44, '[Pd+2]': 45, 'P': 46, '[Al+3]': 47, '[Li+]': 48,
              '[Cs+]': 49, '[NH4+]': 50, '[N-]': 51, '[H]': 52, '[Pd]': 53, '[SiH]': 54, '[Si]': 55, '[Cu+2]': 56, '5': 57, '[Sn]': 58,
              '[Cu]': 59, '[PH]': 60, '[Br-]': 61, '[BH-]': 62, '[BH3-]': 63, '[F-]': 64, '[C-]': 65, '[NH2+]': 66, '6': 67, '[Hg]': 68,
              '[Mn]': 69, '[Fe]': 70, '[Ca+2]': 71, '[se]': 72, '[Cr]': 73, '[nH+]': 74, '[Mg+2]': 75, '[Zn+2]': 76, '[I+3]': 77, '7': 78,
              '8': 79, '[P+]': 80, '[Se]': 81, '[Pb]': 82, '[Ti+4]': 83, '[Mg]': 84, '[n+]': 85, '[Pt]': 86, '[SH]': 87, '[Xe]': 88,
              '[NH+]': 89, '[n-]': 90, '[Rh+]': 91, '[SH-]': 92, '[PH2]': 93, '[Zn+]': 94, '[B+3]': 95, '[S+]': 96, '[Ag]': 97, '[Ru]': 98,
              '[Ce+3]': 99, '[Ce]': 100, '[IH2]': 101, '[O+]': 102, '[B-]': 103, '[SnH]': 104, '[Cl+3]': 105, '[KH]': 106, '[Pd-2]': 107,
              '[AlH]': 108, '[Ba+2]': 109, '[Mn+2]': 110, '[Co]': 111, '[Ti+3]': 112, '[SiH3]': 113, '[NH-]': 114, '[Zr+2]': 115,
              '[Zr+]': 116, '[NH2-]': 117, '[B+]': 118, '[Ag+]': 119, '[Si-]': 120, '[CH-]': 121, '[Cu+]': 122, '[Sb]': 123, '[As]': 124,
              '[Os]': 125, '[CH3]': 126, '[Yb+3]': 127, 'b': 128, '[Ce+4]': 129, '[AlH4-]': 130, '[Rh]': 131, '[Nd+3]': 132, '[PH4+]': 133,
              '[Ni]': 134, '[Fe+2]': 135, '[Sc+3]': 136, '[SiH2]': 137, '[I+2]': 138, '[Ca]': 139, '[s+]': 140, '[GeH]': 141, '[Ge]': 142,
              '[Fe+3]': 143, '[SiH4]': 144, '[Se-]': 145, '[Ti]': 146, '[Ag+2]': 147, '[PH+]': 148, '[NH3+]': 149, '[H+]': 150,
              '[Rh+3]': 151, '[Rh+2]': 152, '[I+]': 153, '[In+3]': 154, '[C+4]': 155, '[Cs]': 156, '[CH2]': 157, '[c-]': 158, '[CH2+]': 159,
              '[Br+2]': 160, '[PH3+]': 161, '[Au-]': 162, '[Au]': 163, '[Nd]': 164, '[Nd+]': 165, '[AlH2-]': 166, '[Zr+4]': 167,
              '[Ag-]': 168, '[P+5]': 169, '[OH]': 170, '[GeH4]': 171, '[Ta+5]': 172, '[Ti+]': 173, '[W]': 174, '[Ti+5]': 175, '[Pt+3]': 176,
              '[PH5]': 177, '[P+3]': 178, '[Te]': 179, '[Pb+4]': 180, '[Sn+2]': 181, '[RuH3]': 182, '[Tl+]': 183, '[Sn+]': 184, '[In]': 185,
              '[Pd+]': 186, '[Pb+2]': 187, '[As+]': 188, '[Fe-3]': 189, '[IH+]': 190, '[CH3+]': 191, '[Sm+3]': 192, '[Cl+2]': 193,
              '[LiH]': 194, '[CH2-]': 195, '[Re]': 196, '[SnH3]': 197, '[SeH-]': 198, '[Ar]': 199, '[Cd+2]': 200, '[PH3]': 201,
              '[Dy+3]': 202, '[Pd-4]': 203, '[Tl]': 204, '[Hf+4]': 205, '[Ir]': 206, '[Bi]': 207, '[Ti+2]': 208, 'p': 209, '[Sn+4]': 210,
              '[Co+2]': 211, '[Mo]': 212, '[Ni+2]': 213, '[Pr+3]': 214, '[Pd-]': 215, '[Zr]': 216, '[o+]': 217, '[Ba]': 218, '[Tl+3]': 219,
              '[SnH4]': 220, '[Yb]': 221, '[Hg+2]': 222, '[AlH3]': 223, '[IH2+]': 224, '[Mn+3]': 225, '[Cr+3]': 226, '[Al+2]': 227,
              '[Sm+2]': 228, '[Sm]': 229, '[C+]': 230, '[BH]': 231, '[SnH2]': 232, '[Ga]': 233, '[Ga+]': 234, '[V+4]': 235, '[IH2+3]': 236,
              '[PdH2]': 237, '[Ta+2]': 238, '[Ta]': 239, '[IH]': 240, '[c+]': 241, '[La]': 242, '[He]': 243, '[Co+]': 244, '[PH2+]': 245,
              '[Tl+2]': 246, '[Sc]': 247, '9': 248, '[Cu+3]': 249, '[La+3]': 250, '[Fe-4]': 251, '[Rb+]': 252, '[Pb+3]': 253, '[Br+]': 254,
              '[Sb+5]': 255, '[Cr+2]': 256, '[Pt+2]': 257, '[Hf]': 258, '[SH3+]': 259, '[N+3]': 260, '[Ga+2]': 261, '[OH+]': 262,
              '[Ru-2]': 263, '%10': 264, '[Cd]': 265, '[Y+3]': 266, '[Hg+]': 267, '[Ru+2]': 268, '[In+]': 269, '[OH2+]': 270, '[Ga+3]': 271,
              '[Bi+3]': 272, '[Co+3]': 273, '[V]': 274, '[Cr+6]': 275, '[PH4]': 276, '[Ni+3]': 277, '[Be+2]': 278, '[V+3]': 279,
              '[Eu+3]': 280, '[Cd+]': 281, '[SeH]': 282, '[Ni+]': 283, '[SH+]': 284, '[CH+]': 285, '[TaH3]': 286, '[Ce+2]': 287,
              '[Zr+3]': 288, '[B+2]': 289, '[Pt-]': 290, '[Sr+2]': 291, '[Sr]': 292, '[SH2]': 293, '[SH2+]': 294}

smi_vocab2 = {}
for smi in smi_vocab.items():
    smi_vocab2[smi[1]] = smi[0]

Words1 = ['Pad', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se',
          'Zn', 'H', 'Cu',
          'Mn', '-', '=', '#', '~', '?', '?5', '.', '>', 'Pd', 'Li', 'Sn', 'Mo', 'Co', 'Cs', 'As']
Words2 = ['Pad', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se',
          'Zn', 'H', 'Cu',
          'Mn', '-', '=', '#', '~', '?', '?5', '.', 'sos', 'eos', 'Pd', 'Li', 'Sn', 'Mo', 'Co', 'Cs', 'As']

src_vocab = {'Pad': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Si': 6, 'P': 7, 'Cl': 8, 'Br': 9, 'Mg': 10,
             'Na': 11, 'Ca': 12, 'Fe': 13, 'Al': 14, 'I': 15, 'B': 16, 'K': 17, 'Se': 18, 'Zn': 19, 'H': 20,
             'Cu': 21, 'Mn': 22, '-': 23, '=': 24, '#': 25, '~': 26, 'Pd': 27, 'Li': 28, 'Sn': 29, 'Mo': 30,
             'Co': 31, 'Cs': 32, 'As': 33, 'Yb': 34, 'Ba': 35, 'Re': 36, 'Cr': 37, 'Ar': 38, 'Dy': 39, 'Ga': 40,
             'In': 41, 'Pr': 42, 'D': 43, 'Sm': 44, 'Cd': 45, 'Hf': 46, 'Ir': 47, 'Rh': 48, 'Zr': 49, 'Ru': 50,
             'Sb': 51, 'V': 52, 'Sr': 53, 'Nd': 54, 'Pb': 55, 'Ta': 56, '.': 57, '>': 58, 'Tc': 59, 'Hg': 60,
             'Ni': 61, 'Pt': 62, 'Ag': 63, 'Ti': 64, 'Au': 65, 'Bi': 66, 'Ce': 67, 'Os': 68, 'Ge': 69, 'W': 70,
             'La': 71, 'Tl': 72, 'Y': 73, 'Te': 74, 'Rb': 75, 'Eu': 76, 'He': 77, 'Xe': 78, 'Gd': 79, 'Sc': 80,
             'Be': 81, 'Er': 82}
src_vocab_size = len(src_vocab)

tgt_vocab = {'Pad': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Si': 6, 'P': 7, 'Cl': 8, 'Br': 9, 'Mg': 10,
             'Na': 11, 'Ca': 12, 'Fe': 13, 'Al': 14, 'I': 15, 'B': 16, 'K': 17, 'Se': 18, 'Zn': 19, 'H': 20,
             'Cu': 21, 'Mn': 22, '-': 23, '=': 24, '#': 25, '~': 26, 'sos': 27, 'eos': 28, 'Pd': 29, 'Li': 30,
             'Sn': 31, 'Mo': 32, 'Co': 33, 'Cs': 34, 'As': 35, 'Yb': 36, 'Ba': 37, 'Re': 38, 'Cr': 39, 'Ar': 40,
             'Dy': 41, 'Ga': 42, 'In': 43, 'Pr': 44, 'D': 45, 'Sm': 46, 'Cd': 47, 'Hf': 48, 'Ir': 49, 'Rh': 50,
             'Zr': 51, 'Ru': 52, 'Sb': 53, 'V': 54, 'Sr': 55, 'Nd': 56, 'Pb': 57, 'Ta': 58, '.': 59, '>': 60,
             'Tc': 61, 'Hg': 62, 'Ni': 63, 'Pt': 64, 'Ag': 65, 'Ti': 66, 'Au': 67, 'Bi': 68, 'Ce': 69, 'Os': 70,
             'Ge': 71, 'W': 72, 'La': 73, 'Tl': 74, 'Y': 75, 'Te': 76, 'Rb': 77, 'Eu': 78, 'He': 79, 'Xe': 80,
             'Gd': 81, 'Sc': 82, 'Be': 83, 'Er': 84}
tgt_vocab_size = len(tgt_vocab)
idx2word = {i: w for i, w in enumerate(tgt_vocab)}


degree_vocab = {'Pad': 20, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11,
                '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19}
degree_vocab_size = len(degree_vocab)


distance_vocab = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
                  '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20,
                  '21': 21, '22': 22, '23': 23, '24': 24, '25': 25, '26': 26, '27': 27, '28': 28, '29': 29, '30': 30,
                  '31': 31, '32': 32, '33': 33, '34': 34, '35': 35, '36': 36, '37': 37, '38': 38, '39': 39, '40': 40,
                  '41': 41, '42': 42, '43': 43, '44': 44, '45': 45, '46': 46, '47': 47, '48': 48, '49': 49, '50': 50,
                  '51': 51, '52': 52, '53': 53, '54': 54, '55': 55, '56': 56, '57': 57, '58': 58, '59': 59, '60': 60,
                  '61': 61, '62': 62, '63': 63, '64': 64, '65': 65, '66': 66, '67': 67, '68': 68, '69': 69, '70': 70,
                  '71': 71, '72': 72, '73': 73, '74': 74, '75': 75, '76': 76, '77': 77, '78': 78, '79': 79, '80': 80,
                  '81': 81, '82': 82, '83': 83, '84': 84, '85': 85, '86': 86, '87': 87, '88': 88, '89': 89, '90': 90,
                  '91': 91, '92': 92, '93': 93, '94': 94, '95': 95, '96': 96, '97': 97, '98': 98, '99': 99, '100': 100,
                  '101': 101, '102': 102, '103': 103, '104': 104, '105': 105, '106': 106, '107': 107, '108': 108,
                  '109': 109, '110': 110, '111': 111, '112': 112, '113': 113, '114': 114, '115': 115, '116': 116, '117': 117,
                  '118': 118, '119': 119, '120': 120, '121': 121, '122': 122, '123': 123, '124': 124, '125': 125, '126': 126,
                  '127': 127, '128': 128, '129': 129, '130': 130, '131': 131, '132': 132, '133': 133, '134': 134, '135': 135,
                  '136': 136, '137': 137, '138': 138, '139': 139, '140': 140, '141': 141, '142': 142, '143': 143, '144': 144,
                  '145': 145, '146': 146, '147': 147, '148': 148, '149': 149, '150': 150, '151': 151, '152': 152, '153': 153,
                  '154': 154, '155': 155, '156': 156, '157': 157, '158': 158, '159': 159, '160': 160, '161': 161, '162': 162,
                  '163': 163, '164': 164, '165': 165, '166': 166, '167': 167, '168': 168, '169': 169, '170': 170, '171': 171,
                  '172': 172, '173': 173, '174': 174, '175': 175, '176': 176, '177': 177, '178': 178, '179': 179, '180': 180,
                  '181': 181, '182': 182, '183': 183, '184': 184, '185': 185, '186': 186, '187': 187, '188': 188, '189': 189,
                  '190': 190, '191': 191, '192': 192, '193': 193, '194': 194, '195': 195, '196': 196, '197': 197, '198': 198,
                  '199': 199}
                  
"""
, '200': 200, '201': 201, '202': 202, '203': 203, '204': 204, '205': 205, '206': 206, '207': 207, '208': 208,
                  '209': 209, '210': 210, '211': 211, '212': 212, '213': 213, '214': 214, '215': 215, '216': 216, '217': 217,
                  '218': 218, '219': 219, '220': 220, '221': 221, '222': 222, '223': 223, '224': 224, '225': 225, '226': 226,
                  '227': 227, '228': 228, '229': 229, '230': 230, '231': 231, '232': 232, '233': 233, '234': 234, '235': 235,
                  '236': 236, '237': 237, '238': 238, '239': 239, '240': 240, '241': 241, '242': 242, '243': 243, '244': 244,
                  '245': 245, '246': 246, '247': 247, '248': 248, '249': 249, '250': 250, '251': 251, '252': 252, '253': 253,
                  '254': 254, '255': 255, '256': 256, '257': 257, '258': 258, '259': 259, '260': 260, '261': 261, '262': 262,
                  '263': 263, '264': 264, '265': 265, '266': 266, '267': 267, '268': 268, '269': 269, '270': 270, '271': 271,
                  '272': 272, '273': 273, '274': 274, '275': 275, '276': 276, '277': 277, '278': 278, '279': 279, '280': 280,
                  '281': 281, '282': 282, '283': 283, '284': 284, '285': 285, '286': 286, '287': 287, '288': 288, '289': 289,
                  '290': 290, '291': 291, '292': 292, '293': 293, '294': 294, '295': 295, '296': 296, '297': 297, '298': 298,
                  '299': 299
"""
distance_vocab_size = len(distance_vocab)

src_len = 300
tgt_len = 300


d_model = 64
d_ff = 512
d_k = d_v = 16
n_layers = 6
n_heads = 4




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

        self.degree_s = nn.Embedding(degree_vocab_size, d_model, padding_idx=20)
        self.MD = nn.Embedding(distance_vocab_size, n_heads, padding_idx=0)

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


class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, smoothing=0.1):

        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0.0 < smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        y_hat = torch.softmax(x, dim=1)
        cross_loss = self.cross_entropy(y_hat, target)
        smooth_loss = -torch.log(y_hat).mean(dim=1)
        loss = self.confidence * cross_loss + self.smoothing * smooth_loss
        return loss.mean()
    def cross_entropy(self, y_hat, y):
        return - torch.log(y_hat[range(len(y_hat)), y])


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.hidden1 = Linear(d_model * 2, 512)
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
            num_t = MA[n][0].tolist().index(0)
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
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns, MLP_output, sum_loss




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

def SymbolAtom2(mol):
    Atomnum = mol.GetNumAtoms()
    for i in range(Atomnum):
        Atom = mol.GetAtomWithIdx(i)
        Atom.SetProp("atomNote", str(i))
    return mol


def SymbolAtom(mol):
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp('molAtomMapNumber',str(i))
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



def lizi(smi):
    t = smi.count("n")
    t2 = smi.count("N")
    paths = list(itertools.product(["n", "[n+]", "[n-]"], repeat=t))
    paths2 = list(itertools.product(["N", "[N+]", "[N-]"], repeat=t2))
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
    return smi

enc_inputs = torch.load("code/90wdata_deal/test_65/enc_in_65.pt")
degree = torch.load("/code/90wdata_deal/test_65/degree_65.pt")
MD = torch.load("/code/90wdata_deal/test_65/MD_65.pt")
dec_inputs = torch.load("/code/90wdata_deal/test_65/enc_in_65.pt")
dec_outputs = torch.load("/code/90wdata_deal/test_65/enc_in_65.pt")
MA = torch.load("/code/90wdata_deal/test_65/MD_65.pt")

batch_size = 100
loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs, degree, MD, MA),
                           batch_size=batch_size, 
                           shuffle=False)
enc_inputs, _, _, degree, MD, _ = next(iter(loader))

class Beamheap:
    def fun(que,x,k=30):
        que.append(x)
        def second(k):
            return k[1]
        que.sort(key=second)
        if len(que)>k:
            que=que[-k:]
        return que

ckpt = torch.load("/code/90wdata_deal/model_80w/2_ckpt_step_10100.pth", map_location='cpu')
model = Transformer()
model.load_state_dict(ckpt["model"])

class beam_search_decoder:
    def __init__(self):
        self.topk = Beamheap.fun

    def forward(self, x0, enc_inputs, degree, MD, max_len=300):

        enc_outputs, enc_self_attns = model.encoder(enc_inputs, degree, MD)

        vector = np.zeros([1, 1, 512])
        vector = torch.LongTensor(vector)
        beams = [([x0], 0.0, vector)]
        for _ in range(max_len):
            que = []
            for x, score, v in beams:
                if x[-1]==28:
                    que = self.topk(que,(x,score,v))
                else:
                    y = torch.from_numpy(np.array(x))
                    dec_input = y.unsqueeze(0)
                    dec_outputs, _, _ = model.decoder(dec_input, enc_inputs, enc_outputs)
                    v = dec_outputs

                    projected_r = model.projection(dec_outputs)
                    projected = projected_r[-1, -1, :]
                    projected = F.log_softmax(projected)
                    output = projected.tolist()
                    for i, o_score in enumerate(output):

                        que = self.topk(que,(x+[i],score + o_score, v))
            beams = que
        return beams

if __name__ == "__main__":
    metas = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']
    data_m = pd.read_csv("/home/user06/90wdata_deal/test65.csv")
    sigmoid = nn.Sigmoid()

    sum_hit = 0
    for i in range(len(enc_inputs)):
        print("section {}".format(i+1))
        S = data_m["Drug"][i]
        src, MA = generate_SA(S)
        print("Source:", ''.join(src))

        num = 0
        m = []
        m_smiles = []
        MA_list = []
        for j in metas:
             meta = data_m[j][i]
             if meta != "N":
                 m_smiles.append(meta)
                 num = num + 1
                 ab_list, MA_tgt  = generate_SA(meta)
                 meta = "".join(ab_list)
                 m.append(meta)
                 MA_list.append(MA_tgt.numpy())
                 print("output_t{}:{}".format(num,meta))
        beam = beam_search_decoder()
        x1 = enc_inputs[i].view(1, -1)
        x2 = degree[i].view(1, -1)
        x3 = MD[i].unsqueeze(0)
        b = beam.forward(27, x1, x2, x3)

        hit_num = 0
        bond_type = ["-", "=", "#", "~"]
        for j in range(len(b)):
            predict_r = [idx2word[n] for n in b[j][0]]
            predict_r_list = predict_r[1:-1]
            predict_r = ''.join(predict_r_list)
            vector = b[j][2]
            num_v = vector.shape[1] - 1
            MA_pred = np.ones([num_v, num_v], dtype=np.int32)*-1
            for x in range(num_v):
                MA_pred[x][x] = 0
            MA_score_0 = np.zeros([num_v, num_v])
            MA_score_1 = np.zeros([num_v, num_v])
            num_bond_list = [0 for x in range(num_v)]
            for x in range(num_v): 
                for y in range(x + 1, num_v):
                    if (predict_r_list[x] in bond_type and predict_r_list[y] not in bond_type) or (predict_r_list[x] not in bond_type and predict_r_list[y] in bond_type):
                        if predict_r_list[x] == "C" and num_bond_list[x] >= 4:
                            continue
                        a1 = vector[0][x + 1]
                        a2 = vector[0][y + 1]
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
                    max_v = [0, [-1,-1]]
                    for y in range(num_v):
                        if MA_score_1[y][x] > max_v[0]:
                            max_v[0] = MA_score_1[y][x]
                            max_v[1] = [y, x]
                    MA_pred[max_v[1][0]][max_v[1][1]] = 1
                    MA_pred[max_v[1][1]][max_v[1][0]] = 1
                if predict_r_list[x] in bond_type and num_bond_list[x] == 3:
                    l_num = []
                    for y in range(x):
                        if MA_pred[y][x] == 1:
                            l_num.append(y)
                    if len(l_num) == 2:
                        if MA_score_1[l_num[0]][x] < MA_score_1[l_num[1]][x]:
                            MA_pred[l_num[0]][x] = -1
                            MA_pred[x][l_num[0]] = -1
                        else:
                            MA_pred[l_num[1]][x] = -1
                            MA_pred[x][l_num[1]] = -1

            print("output_p{}:{}".format(j+1,predict_r))
            try :
                smi_pred = seq_pain(MA_pred, predict_r_list)
                print("smiles_p:", smi_pred)
                smi_pred = lizi(smi_pred)
                for k in m_smiles:
                    t_smiles = Chem.MolFromSmiles(k)
                    p_smiles = Chem.MolFromSmiles(smi_pred)
                    mol = SymbolAtom(p_smiles)
                    if DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(t_smiles), Chem.RDKFingerprint(p_smiles)) == 1:
                        hit_num = hit_num + 1
            except (OverflowError, KeyError, AttributeError, IndexError, RuntimeError, ValueError) as e:
                print("The predicted metabolites did not conform to the rules！")
        print(hit_num)
        sum_hit = sum_hit + hit_num
    print(sum_hit)
