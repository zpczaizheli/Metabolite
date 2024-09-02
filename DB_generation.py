import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import Chem
import copy

Words1 = ['Pad', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn' ,'H', 'Cu', 'Mn', '-', '=', '#', '~', '?', '?5']
Words2 = ['Pad', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn' ,'H', 'Cu', 'Mn', '-', '=', '#', '~', '?', '?5', 'sos', 'eos']

src_vocab = {'Pad': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Si': 6, 'P': 7, 'Cl': 8, 'Br': 9, 'Mg': 10, 'Na': 11, 'Ca': 12, 'Fe': 13, 'Al': 14, 'I': 15, 'B': 16, 'K': 17, 'Se': 18, 'Zn': 19, 'H': 20, 'Cu': 21, 'Mn': 22, '-': 23, '=': 24, '#': 25, '~': 26, '?': 27, '?5': 28}
tgt_vocab = {'Pad': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Si': 6, 'P': 7, 'Cl': 8, 'Br': 9, 'Mg': 10, 'Na': 11, 'Ca': 12, 'Fe': 13, 'Al': 14, 'I': 15, 'B': 16, 'K': 17, 'Se': 18, 'Zn': 19, 'H': 20, 'Cu': 21, 'Mn': 22, '-': 23, '=': 24, '#': 25, '~': 26, '?': 27, '?5': 28, 'sos': 29, 'eos': 30}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}


def GetSequenceFromSmiles(Smiles):
    String = ''
    Mol = Chem.MolFromSmiles(Smiles)
    AtomsNum = Mol.GetNumAtoms()
    for indx1 in range(AtomsNum):
        Atom = Mol.GetAtomWithIdx(indx1)
        AtomSymble = Atom.GetSymbol()
        String += AtomSymble
        for indx2 in range(indx1-1, -1, -1):
            Bond = Mol.GetBondBetweenAtoms(indx1 ,indx2)
            if str(Bond) == 'None':
                String += '?'
                continue
            else:
                BondType = Bond.GetBondType()
                if str(BondType) == 'SINGLE':
                    String += '-'
                elif str(BondType) == 'DOUBLE':
                    String += '='
                elif str(BondType) == 'TRIPLE':
                    String += '#'
                elif str(BondType) == 'AROMATIC':
                    String += '~'
    return String


def my_tokenizer(s):
    L = []
    s = s.replace("Si", "1")
    s = s.replace("Cl", "2")
    s = s.replace("Br", "3")
    s = s.replace("Mg", "4")
    s = s.replace("Na", "5")
    s = s.replace("Ca", "6")
    s = s.replace("Fe", "7")
    s = s.replace("Al", "8")
    s = s.replace("Se", "9")
    s = s.replace("Zn", "x")
    s = s.replace("Cu", "y")
    s = s.replace("Mn", "w")
    for i in s:
        L.append(i)
    L = ['Si' if i == '1' else i for i in L]
    L = ['Cl' if i == '2' else i for i in L]
    L = ['Br' if i == '3' else i for i in L]
    L = ['Mg' if i == '4' else i for i in L]
    L = ['Na' if i == '5' else i for i in L]
    L = ['Ca' if i == '6' else i for i in L]
    L = ['Fe' if i == '7' else i for i in L]
    L = ['Al' if i == '8' else i for i in L]
    L = ['Se' if i == '9' else i for i in L]
    L = ['Zn' if i == 'x' else i for i in L]
    L = ['Cu' if i == 'y' else i for i in L]
    L = ['Mn' if i == 'w' else i for i in L]

    return L


data = pd.read_csv(r"/home/user06/model/data/uspto50k.csv")
uni = pd.DataFrame(data=None, columns=["React", "Product"])
k = 0
for i in range(49924):
    if data['react2'][i] is np.nan:
        react = data.at[i, 'react1']
        product = data.at[i, 'new_product']
        uni.loc[k] = [react, product]
        k = k + 1


def del_null(smi):
    smi_del_tail = []
    flag = True
    for j in range(len(smi) - 1, -1, -1):
        if smi[j] in Bond_list:
            flag = False
        if smi[j] in Atom_list:
            flag = True
        if smi[j] != '?':
            smi_del_tail.append(smi[j])
        elif smi[j] == '?':
            if flag == False:
                smi_del_tail.append('?')
    smi_del_tail.reverse()

    num_null = 0
    smi_del = []
    for k in smi_del_tail:
        smi_del.append(k)
        if k != '?':
            num_null = 0
        if k == '?':
            num_null = num_null + 1
        if num_null > 0 and num_null % 5 == 0:
            smi_del = smi_del[0:len(smi_del) - 5]
            smi_del.append('?5')
    return smi_del


n = 0
L = 0
len_200 = 0
len_100 = 0
len_300 = 0
len_400 = 0
len_500 = 0

uni_100 = pd.DataFrame(data=None, columns=["React", "Product"])
Bond_list = ['-', '=', '#', '~']
Atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn',
             'H', 'Cu', 'Mn']
max_Rlen = 0
max_Plen = 0
Sentences = []

# 14533
for i in range(2):
    n = n + 1
    if n == 10576:
        continue
    product = uni.at[i, 'Product']
    Sequence1 = GetSequenceFromSmiles(product)
    Psmi = my_tokenizer(Sequence1)

    react = uni.at[i, 'React']
    Sequence2 = GetSequenceFromSmiles(react)
    Rsmi = my_tokenizer(Sequence2)

    if len(Rsmi) > L:
        L = len(Rsmi)
    if len(Psmi) > L:
        L = len(Psmi)
    if len(Rsmi) <= 200 and len(Psmi) <= 200:
        len_200 = len_200 + 1
    if len(Rsmi) <= 100 and len(Psmi) <= 100:
        uni_100.loc[len_100] = [Rsmi, Psmi]
        len_100 = len_100 + 1
    if len(Rsmi) <= 300 and len(Psmi) <= 300:
        len_300 = len_300 + 1
    if len(Rsmi) <= 400 and len(Psmi) <= 400:
        len_400 = len_400 + 1
    if len(Rsmi) <= 500 and len(Psmi) <= 500:
        len_500 = len_500 + 1

    Rsmi_del = del_null(Rsmi)
    Psmi_del = del_null(Psmi)

    if len(Rsmi_del) < 301:
        D_V = 301 - len(Rsmi_del)
        for i in range(D_V):
            Rsmi_del.append("Pad")
    if len(Psmi_del) < 313:
        D_V = 313 - len(Psmi_del)
        for i in range(D_V):
            Psmi_del.append("Pad")

    Psmi_del_dec_input = copy.deepcopy(Psmi_del)
    Psmi_del_dec_input.insert(0, 'sos')
    Psmi_del_dec_output = copy.deepcopy(Psmi_del)
    Psmi_del_dec_output.append('eos')
    enc_input = " ".join(Rsmi_del)
    dec_input = " ".join(Psmi_del_dec_input)
    dec_output = " ".join(Psmi_del_dec_output)
    sentence = []
    sentence.append(enc_input)
    sentence.append(dec_input)
    sentence.append(dec_output)
    Sentences.append(sentence)
