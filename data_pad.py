import pandas as pd
import torch
import copy

data1 = pd.read_csv(r"/home/user06/model/data/pre-train/37w_SDD.csv")
data2 = pd.read_csv(r"/home/user06/model/data/pre-train/37w_SA.csv")

df1 = pd.DataFrame(data=None, columns=["enc_inputs", "dec_inputs", "dec_outputs"])
df2 = pd.DataFrame(data=None, columns=["degree"])


max_s = 132
max_t = 106
max_degree = 6
max_dist = 80

for i in range(len(data1.index)): 
    print(i)

    s = data1["enc_inputs"][i]
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace("'", "")
    s = s.replace(",", "")
    s = s.split(" ")
    d = data1["degree_s"][i]
    d = d.replace("[", "")
    d = d.replace("]", "")
    d = d.replace(",", "")
    d = d.split(" ")

    if len(s) <= max_s:
        D_V = max_s - len(s)
        for j in range(D_V):
            s.append("Pad")
            d.append("pad")

    t = data2["dec_inputs"][i]
    t = t.replace("[", "")
    t = t.replace("]", "")
    t = t.replace("'", "")
    t = t.replace(",", "")
    t = t.split(" ")

    t_in = copy.deepcopy(t)
    t_out = copy.deepcopy(t)
    t_in.insert(0, 'sos')
    t_out.append('eos')

    if len(t) <= max_t:
        D_V = max_t - len(t)
        for j in range(D_V):
            t_in.append("Pad")
            t_out.append("pad")

    enc_input = " ".join(s)
    dec_input = " ".join(t_in)
    dec_output = " ".join(t_out)
    degree = " ".join(d)

    df1.loc[len(df1.index)] = [enc_input, dec_input, dec_output]
    df2.loc[len(df2.index)] = [degree]

df1.to_csv(r"/home/user06/model/data/pre-train/enc_dec_in_dec_out.csv")
df2.to_csv(r"/home/user06/model/data/pre-train/degree_enc.csv")