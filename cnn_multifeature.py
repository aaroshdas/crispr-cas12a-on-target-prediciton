import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
import cnn_helper

import numpy as np

import cnn_models
import data_loader




dataset_path = "./datasets/new_features/"
train_path = "NF_Kim_2018_Train.csv"
test_path = "NF_Kim_2018_Test.csv"


COMBINED_DF= data_loader.filter_df_new_features(pd.read_csv(dataset_path + train_path))
TEST_DF= data_loader.filter_df_new_features(pd.read_csv(dataset_path + test_path))



print(COMBINED_DF.head())

TARGET_MEAN = np.load("./weights/target_mean.npy")
TARGET_STD  = np.load("./weights/target_std.npy")


COMBINED_DF['Indel frequency'] = (COMBINED_DF['Indel frequency'] - TARGET_MEAN) / TARGET_STD
TEST_DF['Indel frequency'] = (TEST_DF['Indel frequency'] - TARGET_MEAN) / TARGET_STD


X_feat = COMBINED_DF[["gc content", "pam_prox_at_frac", "pos18_is_c"]].values.astype(np.float32)

X_feat_test = TEST_DF[["gc content", "pam_prox_at_frac", "pos18_is_c"]].values.astype(np.float32)


print(COMBINED_DF.head())
print("total samples", len(COMBINED_DF))

train_sequences = COMBINED_DF["Input seq"].values
raw_x_vals = np.array([cnn_helper.convert_to_one_hot(seq, ) for seq in train_sequences])

raw_y_vals = COMBINED_DF["Indel frequency"].values.astype(float)


def train_model(x_train, x_feat_train, y_train, x_val, x_feat_val, y_val, epochs_):

    model = cnn_models.build_residual_multi_feature_cnn()
    early_stopping= EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr= ReduceLROnPlateau(patience=5)
    
    history = model.fit([x_train, x_feat_train], y_train, epochs=epochs_, batch_size =32, validation_data=([x_val, x_feat_val], y_val), callbacks=[early_stopping, reduce_lr])
    
    return model,history


x_seq_train, x_seq_val, x_feat_train, x_feat_val, y_train, y_val = train_test_split(raw_x_vals, X_feat, raw_y_vals, test_size=0.15)


model, history = train_model(x_seq_train, x_feat_train, y_train, x_seq_val, x_feat_val,y_val,60)

model.save("./weights/multi_feature/weights.keras")


cnn_helper.graph_model_history(history, "multi_feature_cnn_graphs/mae_model_history.png", "mae")

cnn_helper.plot_multi_feature_prediction(model, TEST_DF, "multi_feature_cnn_graphs/predictions_plot.png")

