import pandas as pd
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

def add_features(df):
    df["gc content"] = df["Context Sequence"].apply(gc_content)
    return df

df_train = add_features(df_train)
print(df_train.head())

df_train.to_csv(dataset_path + train_path)