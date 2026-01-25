
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
 


def plot_predictions(pred_method, pred_vals, y_vals, path):
    temp_x_vals = np.linspace(0, len(y_vals)-1, len(y_vals))

    np_y_vals = np.array(y_vals)/100
    np_pred_vals = np.array(pred_vals)/100

    rmse = np.sqrt(np.mean((np_y_vals -np_pred_vals)**2))
    mae = np.mean(np.abs(np_y_vals - np_pred_vals))
    rho, p_value = spearmanr(np_y_vals, np_pred_vals)

    
    
    plt.figure(figsize=(15, 5))
    plt.plot(temp_x_vals, y_vals, label='actual', linestyle='--', color='blue')
    plt.plot(temp_x_vals, pred_vals, label='preds', linestyle='-', color='red')
    plt.ylabel(f'normalized indel freq')
    plt.xlabel(f'RMSE {rmse: .4f} | MAE {mae: .4f} | spearman rho {rho: .4f} | p-val {p_value:.4e}')
    plt.savefig(path)
    print(f'{pred_method} |  RMSE {rmse: .4f} | MAE {mae: .4f} | spearman rho {rho: .4f} | p-val {p_value:.4e} \n')
    results_df[pred_method] = [round(rmse, 4), round(mae,4), round(rho,4)]


df = pd.read_csv('./quickr_data/full_results.csv')

results_df = pd.DataFrame()

for i in df.columns:
    if i != "Score":
        plot_predictions(i, df[i].values, df["Score"].values, f'./quickr_data/graphs/{i}_graph.png')

results_df.to_csv("./quickr_data/summary_results.csv", sep='\t', index =False)