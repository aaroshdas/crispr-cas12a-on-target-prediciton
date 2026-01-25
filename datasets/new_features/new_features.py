import pandas as pd
import numpy as np
old_dataset_path = "./datasets/"
old_train_path = "Kim_2018_Train.csv"
old_test_path = "Kim_2018_Test.csv"

dataset_path = "./datasets/new_features/"
train_path = "NF_Kim_2018_Train.csv"
test_path = "NF_Kim_2018_Test.csv"

df_train = pd.read_csv(old_dataset_path + old_train_path)
df_test = pd.read_csv(old_dataset_path + old_test_path)



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


df_train = add_features(df_train)
df_test = add_features(df_test)

print(df_train.head())

df_train.to_csv(dataset_path + train_path)
df_test.to_csv(dataset_path + test_path)    