#multi task learning
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
import multitask_cnn_helper
import cnn_helper
import numpy as np

import multitask_models
from data_loader import filter_df


#load data

dataset_path = "./datasets/"
train_path = "Kim_2018_Train.csv"
test_path = "Kim_2018_Test.csv"


COMBINED_DF= filter_df(pd.read_csv(dataset_path + train_path))
TEST_DF= filter_df(pd.read_csv(dataset_path + test_path))

TARGET_MEAN = np.load("./weights/target_mean.npy")
TARGET_STD  = np.load("./weights/target_std.npy")

COMBINED_DF['Indel frequency'] = (COMBINED_DF['Indel frequency'] - TARGET_MEAN) / TARGET_STD
TEST_DF['Indel frequency'] = (TEST_DF['Indel frequency'] - TARGET_MEAN) / TARGET_STD

print(COMBINED_DF.head())
print("total samples", len(COMBINED_DF))

train_sequences = COMBINED_DF["Input seq"].values
raw_x_vals = np.array([cnn_helper.convert_to_one_hot(seq, ) for seq in train_sequences])
raw_y_vals = COMBINED_DF["Indel frequency"].values.astype(float)

x_train, x_val, y_train, y_val = train_test_split(raw_x_vals, raw_y_vals, test_size=0.15)



# median = np.median(y_train)
y_train_binary = (y_train > 0.0).astype(np.float32)
y_val_binary   = (y_val > 0.0).astype(np.float32)


# binary_labels, sorted_index = calc_new_features()

def train_model(epochs_):
    #model = cnn_models.load_standard_model(x_train)
    model = multitask_models.load_multitask_model(x_train)

    #overwrites compiler
    early_stopping= EarlyStopping(patience=10, restore_best_weights=True, monitor="val_regression_mae", mode="min")
    reduce_lr= ReduceLROnPlateau(patience=5, monitor="val_regression_mae", mode="min")
    history = model.fit(
       x_train,
        {
            "regression": y_train,
            "binary": y_train_binary
        },
        validation_data=
        (
            x_val,
            {
                "regression": y_val,
                "binary": y_val_binary
            }
        ),
        epochs=epochs_, 
        batch_size =32, 
        callbacks=[early_stopping, reduce_lr]
    )
    return model,history

#temp_k_fold_val(raw_x_vals, raw_y_vals, x_val, y_val, train_model, epochs)

model, history = train_model(60)

model.save("./weights/mtl_cnn_model.keras")

# pairwise ranking model
def get_pairs(x, y, total_num_pairs, min_diff):
    xi =[]
    xj=[] 
    labels = []

    while len(xi) < total_num_pairs:
        i, j = np.random.randint(0, len(y), size=2)
        if abs(y[i] - y[j]) < min_diff:
            continue
        xi.append(x[i])
        xj.append(x[j])
        if(y[i] > y[j]):
            labels.append(1.0)
        else:
            labels.append(0.0)

    return np.array(xi), np.array(xj), np.array(labels)



rank_model = multitask_models.build_ranking_model(model)
rank_model.summary()
for k in range(20):
    xi, xj, y_rank = get_pairs(x_train, y_train,256,0.2)
    rank_model.train_on_batch([xi, xj], y_rank)
    print(f'epoch: {k}')
print("ranking model trained")

multitask_cnn_helper.mt_graph_model_history(history, "mtl_graphs/mae_model_history.png", "mae")
multitask_cnn_helper.mt_plot_predictions(model, len(TEST_DF), TEST_DF, "mtl_graphs/predictions_plot.png")
