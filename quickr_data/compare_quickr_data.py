
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

enpam_data = [0.46800952, 0.45909767, 0.25652697, 0.23232431, 0.42940507, 0.26161265, 
              0.42571779, 0.44305409, 0.20069877, 0.32739208, 0.26335586, 0.42028761, 
              0.34322143, 0.32852092, 0.22314231, 0.47593864, 0.49661462, 0.24428711, 
              0.28206996, 0.40618609, 0.22899019]

cpf1_data = [0.5003338241577149, 0.15581436157226564, 0.48654521942138673, 
             0.025048866271972656, 0.5649512481689453, 0.690278549194336, 
             0.9873972320556641, 0.8700440979003906, 0.3949498748779297, 
             0.5780853271484375, 0.4511328125, 0.7987564086914063, 0.4727542495727539, 
             0.38980392456054686, 0.547983512878418, 0.6068165588378907, 0.7763652038574219, 
             0.7540806579589844, 0.6130593490600585, 0.7727817535400391, 0.43420654296875]

plot_predictions(pred, true_val, './quickr_data/prediction_graph.png')
plot_predictions(deepL, true_val, './quickr_data/pred_rescaled_graph.png')
plot_predictions(enpam_data, true_val, './quickr_data/pred_enpam_graph.png')
plot_predictions(cpf1_data, true_val, './quickr_data/pred_cpf1_graph.png')