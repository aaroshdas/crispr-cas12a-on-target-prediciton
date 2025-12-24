import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
import cnn_helper

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

train_sequences = COMBINED_DF["Input seq"].values
raw_x_vals = np.array([cnn_helper.convert_to_one_hot(seq, ) for seq in train_sequences])

raw_y_vals = COMBINED_DF["Indel frequency"].values.astype(float)


def train_model(x_train, y_train, x_val, y_val, epochs_):
    #model = cnn_models.load_standard_model(x_train)
    model = cnn_models.load_residual_model(x_train)

    #overwrites compiler
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['root_mean_squared_error','mae'])
    early_stopping= EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr= ReduceLROnPlateau(patience=5)
    history = model.fit(x_train, y_train, epochs=epochs_, batch_size =32, validation_data=(x_val, y_val), callbacks=[early_stopping, reduce_lr])
    return model,history

cnn_helper.temp_k_fold_val(raw_x_vals, raw_y_vals, train_model, 60)

# training of model to save
x_train, x_val, y_train, y_val = train_test_split(raw_x_vals, raw_y_vals, test_size=0.15)

model, history = train_model(x_train, y_train, x_val, y_val,60)
model.save("./weights/cnn_model.keras")


cnn_helper.graph_model_history(history, "cnn_graphs/mae_model_history.png", "mae")
cnn_helper.plot_predictions(model, len(TEST_DF), TEST_DF, "cnn_graphs/predictions_plot.png")


# graph_model_history(history, "cnn_graphs/rmse_model_history.png", "root_mean_squared_error")
#plot_predictions(model, 100, COMBINED_DF, "cnn_graphs/predictions_plot.png")
#make_prediction(model, "AGCGTTTAAAAAACATCGAACGCATCTGCTGCCT", TARGET_STD, TARGET_MEAN) #14.711302
