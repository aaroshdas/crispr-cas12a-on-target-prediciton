
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
 

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

    np_y_vals = np.array(temp_y_vals)
    np_pred_vals = np.array(temp_pred_vals)

    rmse = np.sqrt(np.mean((np_y_vals -np_pred_vals)**2))
    mae = np.mean(np.abs(np_y_vals - np_pred_vals))
    rho, p_value = spearmanr(np_y_vals, np_pred_vals)

    
    
    plt.figure(figsize=(15, 5))
    plt.plot(temp_x_vals, temp_y_vals, label='actual', linestyle='--', color='blue')
    plt.plot(temp_x_vals, temp_pred_vals, label='preds', linestyle='-', color='red')
    plt.ylabel(f'normalized indel freq')
    plt.xlabel(f'RMSE {rmse: .4f} | MAE {mae: .4f} | spearman rho {rho: .4f} | p-val {p_value:.4e}')
    plt.savefig(path)
    print(f'RMSE {rmse: .4f} | MAE {mae: .4f} | spearman rho {rho: .4f} | p-val {p_value:.4e}')

def make_prediction(model, seq, TARGET_STD, TARGET_MEAN):
    temp_batch_seq = convert_to_one_hot(seq)[np.newaxis, ...]
    pred = model.predict(temp_batch_seq, verbose=False)
    print(f'\n {pred * TARGET_STD + TARGET_MEAN}')


def temp_k_fold_val(raw_x_vals, raw_y_vals, train_model, epochs, splits=4):
    kf = KFold(n_splits=splits, shuffle=True, random_state=42)
    fold_histories = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(raw_x_vals)):
        kf_x_train, kf_x_val = raw_x_vals[train_idx], raw_x_vals[val_idx]
        kf_y_train, kf_y_val = raw_y_vals[train_idx], raw_y_vals[val_idx]

        kf_model, kf_history = train_model(kf_x_train, kf_y_train, kf_x_val, kf_y_val, epochs)
        
        print(f"fold {fold + 1}/{splits}")
        fold_histories.append(kf_history)
        res = kf_model.evaluate(kf_x_val, kf_y_val, verbose=0)
        fold_metrics.append(res)
        print(f'loss {res[0]:.4f} - mae: {res[1]:.4f}')
        print("")
    print("k-fold results")
    for k in range(len(fold_metrics)):
        print(f"fold {k + 1} | loss: {fold_metrics[k][0]:.4f} - mae: {fold_metrics[k][1]:.4f}")
