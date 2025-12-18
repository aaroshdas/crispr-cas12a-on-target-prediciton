

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from cnn_helper import *

def predict_sequence_xg_boost(seq, embedding_model, xgb_model):
    onehot = convert_to_one_hot(seq)
    onehot = np.expand_dims(onehot, axis=0)

    embed = embedding_model.predict(onehot)
    pred_norm = xgb_model.predict(embed)[0]
    
    return pred_norm

def plot_predictions_xg_boost(total_x_vals, COMBINED_DF, path, embedding_model, xgb_model):
    temp_y_vals = []
    temp_pred_vals = []
    temp_x_vals = []
    for i in range(total_x_vals):
        temp_y_vals.append(COMBINED_DF.iloc[i, 1])
        temp_x_vals.append(i)
        seq = COMBINED_DF.iloc[i,  0]
        temp_batch_seq = convert_to_one_hot(seq)[np.newaxis, ...]
    
        pred = predict_sequence_xg_boost(temp_batch_seq, embedding_model, xgb_model)
        temp_pred_vals.append(pred)
    
    plt.figure(figsize=(15, 5))
    plt.plot(temp_x_vals, temp_y_vals, label='actual', linestyle='--', color='blue')
    plt.plot(temp_x_vals, temp_pred_vals, label='preds', linestyle='-', color='red')
    plt.ylabel('normalized indel freq')
    plt.savefig(path)
