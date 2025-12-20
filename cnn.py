#cnn 
#conda activate cc12on
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from cnn_helper import *

import numpy as np

import cnn_models



dataset_path = "./datasets/"
train_path = "Kim_2018_Train.csv"
test_path = "Kim_2018_Test.csv"


def filter_df(df):
    df = df.drop(columns=["50 bp synthetic target and target context sequence","20 bp guide sequence (5' to 3')","Indel frequency (% Background)","Indel read count (Background)","Total read count (Background)","Indel freqeuncy (Cpf1 delivered %)", "Indel read count (Cpf1 delivered)","Total read count (Cpf1 delivered)"], axis=1)
    df.rename(columns={"Context Sequence": "Input seq"}, inplace=True)
    df = df[df['Indel frequency'] >= -1]
    df['Indel frequency'] = df['Indel frequency'].clip(lower=0)
    
    #old normalization, diff one used
    #df['Indel frequency_norm'] = df['Indel frequency'] / 100.0

    return df

temp_train_df= filter_df(pd.read_csv(dataset_path + train_path))

temp_test_df= filter_df(pd.read_csv(dataset_path + test_path))

COMBINED_DF = pd.concat([temp_train_df,temp_test_df])

print(COMBINED_DF.head())

TARGET_MEAN = COMBINED_DF['Indel frequency'] .mean()
TARGET_STD = COMBINED_DF['Indel frequency'].std()

#normalize indel frequnecy
COMBINED_DF['Indel frequency'] = (COMBINED_DF['Indel frequency'] - TARGET_MEAN) / TARGET_STD

print(COMBINED_DF.head())
print("total samples", len(COMBINED_DF))

# print(convert_to_one_hot(test_df.loc[0, "Input seq"]))

train_sequences = COMBINED_DF["Input seq"].values
raw_x_vals = np.array([convert_to_one_hot(seq, ) for seq in train_sequences])

raw_y_vals = COMBINED_DF["Indel frequency"].values.astype(float)


x_train, x_val, y_train, y_val = train_test_split(raw_x_vals, raw_y_vals, test_size=0.15)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)


#model = cnn_models.load_standard_model(x_train)
model = cnn_models.load_residual_model(x_train)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['root_mean_squared_error','mae'])
model.summary()


#make_prediction(model, "AGCGTTTAAAAAACATCGAACGCATCTGCTGCCT", TARGET_STD, TARGET_MEAN) #14.711302


history = model.fit(x_train, y_train, epochs=30, batch_size =32, validation_data=(x_val, y_val))

model.save("cnn_model.keras")

#make_prediction(model, "AGCGTTTAAAAAACATCGAACGCATCTGCTGCCT",TARGET_STD, TARGET_MEAN)




graph_model_history(history, "cnn_graphs/mae_model_history.png", "mae")

graph_model_history(history, "cnn_graphs/rmse_model_history.png", "root_mean_squared_error")


plot_predictions(model, 100, COMBINED_DF, "cnn_graphs/predictions_plot.png")
