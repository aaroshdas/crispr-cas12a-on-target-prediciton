
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
 

def plot_predictions(pred_vals, y_vals, path):
    temp_x_vals = np.linspace(0, len(y_vals)-1, len(y_vals))

    np_y_vals = np.array(y_vals)
    np_pred_vals = np.array(pred_vals)

    rmse = np.sqrt(np.mean((np_y_vals -np_pred_vals)**2))
    mae = np.mean(np.abs(np_y_vals - np_pred_vals))
    rho, p_value = spearmanr(np_y_vals, np_pred_vals)

    
    
    plt.figure(figsize=(15, 5))
    plt.plot(temp_x_vals, y_vals, label='actual', linestyle='--', color='blue')
    plt.plot(temp_x_vals, pred_vals, label='preds', linestyle='-', color='red')
    plt.ylabel(f'normalized indel freq')
    plt.xlabel(f'RMSE {rmse: .4f} | MAE {mae: .4f} | spearman rho {rho: .4f} | p-val {p_value:.4e}')
    plt.savefig(path)
    print(f'RMSE {rmse: .4f} | MAE {mae: .4f} | spearman rho {rho: .4f} | p-val {p_value:.4e}')




df = pd.read_csv('./quickr_data/quickr_data.csv')
print(df.columns)
df["pred"] = df["pred"] /100
pred = df["pred"].tolist()
deepL = df["rescaled_pred"].tolist()
true_val = df["val"].tolist()


plot_predictions(pred, true_val, './quickr_data/prediction_graph.png')
plot_predictions(deepL, true_val, './quickr_data/pred_rescaled_graph.png')