import pandas as pd
import numpy as np

# old_dataset_path = "./datasets/"
# old_data_path = "Kim_2018_Train.csv"

# dataset_path = "./datasets/new_features/"
# data_path = "NF_Kim_2018_Train.csv"

old_dataset_path = "./quickr_data/"
old_data_path = "raw_quickr_seqs.csv"

dataset_path = "./quickr_data/"
data_path = "quickr_seqs_new_features.csv"

raw_data_df = pd.read_csv(old_dataset_path + old_data_path)

data_df = pd.DataFrame()

shortened_seqs = []
for i in raw_data_df["Context Sequence"]:
    shortened_seqs.append(i[:34])

data_df["Context Sequence"] = shortened_seqs


def gc_content(seq):
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq)


# no point really given all data is tttv
# def pam_is_tttn(seq, pam_start=4, pam_len=4):
#     pam = seq[pam_start:pam_start+pam_len]
#     return int(pam[:3] == "TTT")

def pam_prox_at_fraction(seq, pam_start=4, pam_len=4, window=10):
    start = pam_start + pam_len
    prox = seq[start:start + window].upper()
    if len(prox) == 0:
        return np.nan
    return (prox.count("A") + prox.count("T")) / len(prox)

def pos18_is_c(seq, pam_start=4, pam_len=4):
    spacer = seq[pam_start + pam_len:].upper()
    if len(spacer) < 18:
        return 0
    return int(spacer[17] == "C")



def add_features(df):
    df["gc content"] = df["Context Sequence"].apply(gc_content)
    df["pam_prox_at_frac"] = df["Context Sequence"].apply(pam_prox_at_fraction)
    df["pos18_is_c"] = df["Context Sequence"].apply(pos18_is_c)

    return df


data_df = add_features(data_df)

print(data_df.head())

data_df.to_csv(dataset_path + data_path)
