#cnn 
#conda activate cc12on
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers # type: ignore
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from cnn_helper import *

import numpy as np

import cnn_models
from data_loader import filter_df


dataset_path = "./datasets/"
train_path = "Kim_2018_Train.csv"
test_path = "Kim_2018_Test.csv"





COMBINED_DF= filter_df(pd.read_csv(dataset_path + train_path))

TEST_DF= filter_df(pd.read_csv(dataset_path + test_path))

# COMBINED_DF = pd.concat([temp_train_df,temp_test_df])

print(COMBINED_DF.head())


TARGET_MEAN = np.load("./weights/target_mean.npy")
TARGET_STD  = np.load("./weights/target_std.npy")


COMBINED_DF['Indel frequency'] = (COMBINED_DF['Indel frequency'] - TARGET_MEAN) / TARGET_STD
TEST_DF['Indel frequency'] = (TEST_DF['Indel frequency'] - TARGET_MEAN) / TARGET_STD


print(COMBINED_DF.head())
print("total samples", len(COMBINED_DF))

# print(convert_to_one_hot(test_df.loc[0, "Input seq"]))

train_sequences = COMBINED_DF["Input seq"].values
raw_x_vals = np.array([convert_to_one_hot(seq, ) for seq in train_sequences])

raw_y_vals = COMBINED_DF["Indel frequency"].values.astype(float)


x_train, x_val, y_train, y_val = train_test_split(raw_x_vals, raw_y_vals, test_size=0.15)

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)



def train_model(x_train, y_train, x_val, y_val, epochs_):
    #model = cnn_models.load_standard_model(x_train)
    model = cnn_models.load_residual_model(x_train)

    #overwrites compiler
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['root_mean_squared_error','mae'])

    history = model.fit(x_train, y_train, epochs=epochs_, batch_size =32, validation_data=(x_val, y_val))
    return model,history

#temp k fold val
def temp_k_fold_val():
    N_SPLITS = 3
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_histories = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(raw_x_vals)):
        kf_x_train, kf_x_val = raw_x_vals[train_idx], raw_x_vals[val_idx]
        kf_y_train, kf_y_val = raw_y_vals[train_idx], raw_y_vals[val_idx]

        kf_model, kf_history = train_model(kf_x_train, kf_y_train, kf_x_val, kf_y_val, 30)
        
        print(f"fold {fold + 1}/{N_SPLITS}")
        fold_histories.append(kf_history)
        res = kf_model.evaluate(x_val, y_val, verbose=0)
        fold_metrics.append(res)
        print(f'loss {res[0]:.4f} - rmse: {res[1]:.4f} - mae: {res[2]:.4f}')
        print("")
    print("k-fold results")
    print(fold_metrics)

#temp_k_fold_val()



#final training of model to save
model, history = train_model(x_train, y_train, x_val, y_val,60)
model.save("./weights/cnn_model.keras")


graph_model_history(history, "cnn_graphs/mae_model_history.png", "mae")
graph_model_history(history, "cnn_graphs/rmse_model_history.png", "root_mean_squared_error")


plot_predictions(model, len(TEST_DF), TEST_DF, "cnn_graphs/predictions_plot.png")



#plot_predictions(model, 100, COMBINED_DF, "cnn_graphs/predictions_plot.png")
#make_prediction(model, "AGCGTTTAAAAAACATCGAACGCATCTGCTGCCT", TARGET_STD, TARGET_MEAN) #14.711302
