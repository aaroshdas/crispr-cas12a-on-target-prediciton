import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models # type: ignore
import xgboost as xgb
from cnn_helper import convert_to_one_hot
from xgb_helper import plot_predictions_xg_boost
from xgboost_ensemble_cnn_models import get_residual_cnn_pooling_index
from data_loader import filter_df

dataset_path = "./datasets/"
test_path = "Kim_2018_Test.csv"

weights_path = "./weights/saved_models/ensemble/"

cnn_model_path = weights_path+ "embeddings_01.keras"
xgb_model_path= weights_path+ "xgb_01.json"

MAX_POOLING_LAYER_INDEX = get_residual_cnn_pooling_index()



test_df = filter_df(pd.read_csv(dataset_path + test_path))[:200]


sequences = test_df["Input seq"].values
x_vals = np.array([convert_to_one_hot(seq) for seq in sequences])

cnn_model = tf.keras.models.load_model(cnn_model_path)

embedding_model = models.Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.layers[MAX_POOLING_LAYER_INDEX].output
)

xgb_model = xgb.XGBRegressor()
xgb_model.load_model(xgb_model_path)

x_embed = embedding_model(x_vals, training=False).numpy()
print("Embedding shape:", x_embed.shape)

y_pred_normalized = xgb_model.predict(x_embed)


def compare_pred(test_df):
    TARGET_MEAN = np.load("./weights/target_mean.npy")
    TARGET_STD  = np.load("./weights/target_std.npy")

    y_pred = y_pred_normalized * TARGET_STD + TARGET_MEAN

    test_df["Pred"] = y_pred

    
    test_df["Indel frequency"] = (test_df["Indel frequency"]- TARGET_MEAN) / TARGET_STD
    
    plot_predictions_xg_boost(len(test_df), test_df, "./cnn_xgb_graphs/tests/test_predictions_plot.png", embedding_model, xgb_model)
    
    test_df["Indel frequency"] = (test_df["Indel frequency"]* TARGET_STD) + TARGET_MEAN
    
    # print(test_df[["Indel frequency", "Pred"]].head())
    print(test_df.head())

compare_pred(test_df)


