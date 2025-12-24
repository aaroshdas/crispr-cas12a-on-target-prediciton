
import matplotlib.pyplot as plt
import numpy as np
import cnn_helper 
from scipy.stats import spearmanr
 

def mt_graph_model_history(history, path, metric):
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
    plt.plot(history.history[f'regression_{metric}'])
    plt.plot(history.history[f'val_regression_{metric}'])
    plt.title(f'model regression {metric}')
    plt.ylabel(f'{metric}')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.savefig(path)

def mt_plot_predictions(model, total_x_vals, COMBINED_DF, path):
    temp_y_vals = []
    temp_pred_vals = []
    temp_x_vals = []
    for i in range(total_x_vals):
        temp_y_vals.append(COMBINED_DF.iloc[i, 1])
        temp_x_vals.append(i)
        seq = COMBINED_DF.iloc[i,  0]
        temp_batch_seq = cnn_helper.convert_to_one_hot(seq)[np.newaxis, ...]
    
        pred = model.predict(temp_batch_seq, verbose=False)
        temp_pred_vals.append(pred["regression"][0][0])

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
