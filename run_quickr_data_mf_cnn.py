import pandas as pd
import numpy as np
import tensorflow as tf
# from tensorflow.keras import models
from cnn_helper import convert_to_one_hot
from scipy.stats import spearmanr
from matplotlib import pyplot as plt


def plot_predictions_from_df(test_df, path):
    y_true = test_df.iloc[:, 1].values.astype(np.float32)
    y_pred = test_df.iloc[:, 2].values.astype(np.float32)
    x_vals = np.linspace(0, len(y_true)-1, len(y_true))

    rmse = np.sqrt(np.mean((y_true -y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    rho, p_value = spearmanr(y_true, y_pred)
    
    plt.figure(figsize=(15, 5))
    plt.plot(x_vals, x_vals, label='actual', linestyle='--', color='blue')
    plt.plot(x_vals, x_vals, label='preds', linestyle='-', color='red')
    plt.ylabel(f'normalized indel freq')
    plt.xlabel(f'RMSE {rmse: .4f} | MAE {mae: .4f} | spearman rho {rho: .4f} | p-val {p_value:.4e}')
    plt.savefig(path)
    print(f'RMSE {rmse: .4f} | MAE {mae: .4f} | spearman rho {rho: .4f} | p-val {p_value:.4e}')

file = './quickr_data/NF_raw_quickr_seqs.csv'

df = pd.read_csv(file)
print(df.head())

#use cnn
weights_path = "./weights/saved_models/cnn/"
cnn_model_path = weights_path+ "save_01.keras"

seqs = df["Context Sequence"].values

def proccess_seqs(seqs):
    procssesed_seqs = []
    for s in seqs:
        new_s = s
        if len(s) > 34:
            new_s= s[:34]
        procssesed_seqs.append(new_s)
    return procssesed_seqs
procssesed_seqs = proccess_seqs(seqs)

df['Context Sequence'] = procssesed_seqs
df.to_csv('./quickr_data/processed_quickr_seqs.csv', index=False)

TARGET_MEAN = np.load("./weights/target_mean.npy")
TARGET_STD  = np.load("./weights/target_std.npy")

x_vals = np.array([convert_to_one_hot(seq) for seq in procssesed_seqs])

model = tf.keras.models.load_model(cnn_model_path)

y_pred_normalized = model.predict(x_vals).squeeze()
y_pred = (y_pred_normalized * TARGET_STD) + TARGET_MEAN

df["Indel frequency Prediction"] = y_pred


for i in y_pred:
    print(f'{i:.4f}')
print(df)
