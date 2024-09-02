import pandas as pd
import numpy as np
from rdkit import Chem
import copy
import pandas as pd
# from indigo import *
import os
import platform
# from ctypes import CDLL, POINTER, c_char_p, c_int
# from indigo import DECODE_ENCODING, IndigoException
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw
# import matplotlib.pyplot as plt
# indigo = Indigo()
from rdkit.Chem import rdChemReactions
import torch
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
# from IPython.display import SVG
# from matplotlib.colors import ColorConverter


data1 = pd.read_csv(r"/home/user06/model/data/pre-train/pre_train_smi.csv")
def generate_SDD(smi):
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

    return MD

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
    return MA

df1 = pd.DataFrame(data=None, columns=["MD"])
df2 = pd.DataFrame(data=None, columns=["MA"])


for i in range(len(data1.index)):
    print(i)
    src = "".join(data1["src"][i].split(" "))
    tgt = "".join(data1["tgt"][i].split(" "))

    MD = generate_SDD(src)
    MA = generate_SA(tgt)

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
        
    df1.loc[len(df1.index)] = [MD2]
    df2.loc[len(df2.index)] = [MA2]

df1.to_csv(r"/home/user06/model/data/pre-train/MD.csv")
df2.to_csv(r"/home/user06/model/data/pre-train/MA.csv")