
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

def convert_to_one_hot(seq):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    one_hot_temp_list= []
    for base in seq:
        one_hot_temp_list.append(mapping.get(base))
    return np.array(one_hot_temp_list)


def graph_model_history(history, path, metric):
    #loss vals
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])

    #mae vals
    plt.subplot(1, 2, 2)
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'model {metric}')
    plt.ylabel(f'{metric}')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.savefig(path)

def plot_predictions(model, total_x_vals, COMBINED_DF, path):
    temp_y_vals = []
    temp_pred_vals = []
    temp_x_vals = []
    for i in range(total_x_vals):
        temp_y_vals.append(COMBINED_DF.iloc[i, 1])
        temp_x_vals.append(i)
        seq = COMBINED_DF.iloc[i,  0]
        temp_batch_seq = convert_to_one_hot(seq)[np.newaxis, ...]
    
        pred = model.predict(temp_batch_seq, verbose=False)
        temp_pred_vals.append(pred[0][0])
    
    plt.figure(figsize=(15, 5))
    plt.plot(temp_x_vals, temp_y_vals, label='actual', linestyle='--', color='blue')
    plt.plot(temp_x_vals, temp_pred_vals, label='preds', linestyle='-', color='red')
    plt.ylabel('normalized indel freq')
    plt.savefig(path)

def make_prediction(model, seq, TARGET_STD, TARGET_MEAN):
    temp_batch_seq = convert_to_one_hot(seq)[np.newaxis, ...]
    pred = model.predict(temp_batch_seq, verbose=False)
    print(f'\n {pred * TARGET_STD + TARGET_MEAN}')