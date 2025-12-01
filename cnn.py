#cnn 
#conda activate cc12on
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

dataset_path = "./datasets/"
train_path = "Kim_2018_Train.csv"
test_path = "Kim_2018_Test.csv"

train_df= pd.read_csv(dataset_path + train_path)
train_df = train_df.drop(columns=["Context Sequence","20 bp guide sequence (5' to 3')","Indel frequency (% Background)","Indel read count (Background)","Total read count (Background)","Indel freqeuncy (Cpf1 delivered %)", "Indel read count (Cpf1 delivered)","Total read count (Cpf1 delivered)"], axis=1)


test_df= pd.read_csv(dataset_path + test_path)
test_df = test_df.drop(columns=["Context Sequence","20 bp guide sequence (5' to 3')","Indel frequency (% Background)","Indel read count (Background)","Total read count (Background)","Indel freqeuncy (Cpf1 delivered %)", "Indel read count (Cpf1 delivered)","Total read count (Cpf1 delivered)"], axis=1)
print(test_df.head())
