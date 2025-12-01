#cnn 
#conda activate cc12on
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

import numpy as np



dataset_path = "./datasets/"
train_path = "Kim_2018_Train.csv"
test_path = "Kim_2018_Test.csv"


def filter_df(df):
    df = df.drop(columns=["50 bp synthetic target and target context sequence","20 bp guide sequence (5' to 3')","Indel frequency (% Background)","Indel read count (Background)","Total read count (Background)","Indel freqeuncy (Cpf1 delivered %)", "Indel read count (Cpf1 delivered)","Total read count (Cpf1 delivered)"], axis=1)
    df.rename(columns={"Context Sequence": "Input seq"}, inplace=True)
    df = df[df['Indel frequency'] >= -1]
    df['Indel frequency'] = df['Indel frequency'].clip(lower=0)
    
    #df['Indel frequency_norm'] = df['Indel frequency'] / 100.0

    return df

temp_train_df= filter_df(pd.read_csv(dataset_path + train_path))

temp_test_df= filter_df(pd.read_csv(dataset_path + test_path))

combined_df = pd.concat([temp_train_df,temp_test_df])

print("total samples", len(combined_df))

print(combined_df.head())

def convert_to_one_hot(seq):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    one_hot_temp_list= []
    for base in seq:
        one_hot_temp_list.append(mapping.get(base))
    return np.array(one_hot_temp_list)

# print(convert_to_one_hot(test_df.loc[0, "Input seq"]))

train_sequences = combined_df["Input seq"].values
raw_x_vals = np.array([convert_to_one_hot(seq) for seq in train_sequences])

raw_y_vals = combined_df["Indel frequency"].values.astype(float)


x_train, x_val, y_train, y_val = train_test_split(raw_x_vals, raw_y_vals, test_size=0.15)
print(x_train.shape, x_val.shape)



