import pandas as pd

import os
import platform
from ctypes import CDLL, POINTER, c_char_p, c_int
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdChemReactions

def SymbolAtom(mol):
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp('molAtomMapNumber',str(i))
    return mol

import pandas as pd
import numpy as np
from rdkit import Chem
import copy
import pandas as pd
import os
import platform
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdChemReactions
import torch
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
data1 = pd.read_csv("/home/user06/model/analyze/Analysis example.csv")


def generate_SDD(smi):
    Mol = Chem.MolFromSmiles(smi)
    AtomsNum = Mol.GetNumAtoms()

    ab_list = []
    degree_list = []
    B_A = []

    for index1 in range(AtomsNum):
        num_input_atom = 0
        Atom = Mol.GetAtomWithIdx(index1)
        AtomSymble = Atom.GetSymbol()
        degree = Atom.GetDegree()
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
                degree_list.append(2)
                if num_input_atom == 0:
                    degree_list.append(degree)
                    ab_list.append(AtomSymble)
                    num_input_atom = num_input_atom + 1

        if num_input_atom == 0:
            degree_list.append(degree)
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
    DM2 = Chem.GetDistanceMatrix(mol)
    DM2 = torch.from_numpy(DM2)
    for i in range(DM2.shape[0]):
        for j in range(DM2.shape[0]):
            if DM2[i][j] > 1000:
                DM2[i][j] = -1

    MD = (torch.ones(num_v, num_v, dtype=torch.int)) * -1

    for i in range(num_v):
        MD[i][i] = 0

    for i in range(AtomsNum):
        for j in range(i + 1, AtomsNum):
            if DM2[i][j] != -1:
                MD[A_list[i]][A_list[j]] = DM2[i][j] * 2
                MD[A_list[j]][A_list[i]] = DM2[i][j] * 2

    for i in range(AtomsNum):
        for j in range(len(B_list)):
            dist = min(DM2[i][B_A[j][0]], DM2[i][B_A[j][1]])
            if dist != -1:
                MD[A_list[i]][B_list[j]] = dist * 2 + 1
                MD[B_list[j]][A_list[i]] = dist * 2 + 1

    for i in range(len(B_list)):
        for j in range(i + 1, len(B_list)):
            dist = min(DM2[B_A[i][0]][B_A[j]][0], DM2[B_A[i][0]][B_A[j]][1], DM2[B_A[i][1]][B_A[j]][0],
                       DM2[B_A[i][1]][B_A[j]][1])
            if dist != -1:
                MD[B_list[i]][B_list[j]] = dist * 2 + 2
                MD[B_list[j]][B_list[i]] = dist * 2 + 2

    return ab_list, degree_list, MD

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

df1 = pd.DataFrame(data=None, columns=["source", "target"])
for i in range(len(data1.index)):
    print(i)
    x = data1["source"][i]
    y = data1["target"][i]
    # try:
    n1, _ = generate_SA(x)
    n2, _ = generate_SA(y)
    if len(n1) <= 131 and len(n2) <= 106:
        df1.loc[len(df1.index)] = [x, y]

df2 = pd.DataFrame(data=None, columns=["Drug", "Degree", "MD"])
max_s = 132
max_t = 106
max_degree = 6
max_dist = 80

for i in range(len(df1.index)):  
    print(i)
    src = df1["source"][i]
    tgt = df1["target"][i]

    s, d, MD = generate_SDD(src)
    t, MA = generate_SA(tgt)
    
    
    if torch.max(MD) <= 80:
        if len(s) <= max_s:
            D_V = max_s - len(s)
            for j in range(D_V):
                s.append("Pad")
                d.append("pad") 
    
        t_in = copy.deepcopy(t)
        t_out = copy.deepcopy(t)
        t_in.insert(0, 'sos')
        t_out.append('eos')
    
        if len(t) <= max_t:
            D_V = max_t - len(t)
            for j in range(D_V):  
                t_in.append("Pad") 
                t_out.append("Pad")
    
        MD1 = MD.numpy().tolist()
        MA1 = MA.numpy().tolist() 
        MD2 = ""
        MA2 = "" 
     
        for j in range(len(MD1)): 
            x = " ".join(list(map(str, MD1[j])))
            if j != 0:
                MD2 = MD2 + " "
            MD2 = MD2 + x
    
        for j in range(len(MA1)):  
            x = " ".join(list(map(str, MA1[j]))) 
            if j != 0:
                MA2 = MA2 + " "
            MA2 = MA2 + x
    
        enc_input = " ".join(s)
        dec_input = " ".join(t_in)
        dec_output = " ".join(t_out)
        d = [str(j) for j in d]
        degree = " ".join(d)

        df2.loc[len(df2.index)] = [enc_input, degree, MD2]
df2.to_csv("/home/user06/model/analyze/example input.csv")