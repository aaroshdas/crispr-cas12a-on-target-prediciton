#xg boost
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, Model # type: ignore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from cnn_helper import *
import numpy as np
import xgboost as xgb  
from xgb_helper import *
import xgboost_ensemble_cnn_models 
from data_loader import filter_df
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

dataset_path = "./datasets/"
train_path = "Kim_2018_Train.csv"
test_path = "Kim_2018_Test.csv"

#csv needs to be formated: Context Sequence, Indel frequency


COMBINED_DF= filter_df(pd.read_csv(dataset_path + train_path))

TEST_DF= filter_df(pd.read_csv(dataset_path + test_path))

# COMBINED_DF = pd.concat([temp_train_df,temp_test_df])

print(COMBINED_DF.head())

TARGET_MEAN = COMBINED_DF['Indel frequency'].mean()
TARGET_STD = COMBINED_DF['Indel frequency'].std()

#temp
np.save("./weights/target_mean.npy", TARGET_MEAN)
np.save("./weights/target_std.npy", TARGET_STD)

TARGET_MEAN = np.load("./weights/target_mean.npy")
TARGET_STD  = np.load("./weights/target_std.npy")


COMBINED_DF['Indel frequency'] = (COMBINED_DF['Indel frequency'] - TARGET_MEAN) / TARGET_STD
TEST_DF['Indel frequency'] = (TEST_DF['Indel frequency'] - TARGET_MEAN) / TARGET_STD


train_sequences = COMBINED_DF["Input seq"].values

raw_x_vals = np.array([convert_to_one_hot(seq, ) for seq in train_sequences])
raw_y_vals = COMBINED_DF["Indel frequency"].values.astype(float)


x_train, x_val, y_train, y_val = train_test_split(raw_x_vals, raw_y_vals, test_size=0.15)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
#(14796, 34, 4) (2612, 34, 4) (14796,) (2612,)




#RESIDUAL CNN
model = xgboost_ensemble_cnn_models.build_residual_cnn(x_train)
#model= xgboost_ensemble_cnn_models.build_standard_cnn(x_train)

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

early_stopping= EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr= ReduceLROnPlateau(patience=5)

history = model.fit(x_train, y_train, epochs=60, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping, reduce_lr])
graph_model_history(history, "cnn_xgb_graphs/cnn_embeddings_history.png", "mae")


model.build(input_shape=(None, x_train.shape[1], 4))
embedding_model = models.Model(inputs=model.inputs, outputs=model.get_layer("embedding").output)
model.save("./weights/cnn_embeddings_model.keras")


x_train_embed = embedding_model(x_train, training=False).numpy()
x_val_embed = embedding_model(x_val, training=False).numpy()

print("embedding shape", x_train_embed.shape)

tnse_embedding_visualization(x_train_embed, y_train)
umap_embedding_visualization(x_train_embed, y_train)

xgb_model = xgb.XGBRegressor(
    n_estimators=850,
    learning_rate=0.02,
    max_depth=4,
    early_stopping_rounds=40,

    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=10.0,
    objective="reg:absoluteerror",
    tree_method="hist"
    )

#calc residuals from cnn
# y_train_cnn = model.predict(x_train).squeeze()
# y_val_cnn = model.predict(x_val).squeeze()
# y_train_residual = y_train - y_train_cnn
# y_val_residual   = y_val - y_val_cnn
# xgb_model.fit(x_train_embed, y_train_residual, eval_set=[(x_val_embed, y_val_residual)], verbose=True)

xgb_model.fit(x_train_embed, y_train, eval_set=[(x_val_embed, y_val)], verbose=True)
xgb_model.save_model("./weights/cnn_xgb_model.json")



plot_predictions_xg_boost(len(TEST_DF), TEST_DF, "cnn_xgb_graphs/predictions_plot.png", embedding_model,xgb_model)

#plot_predictions_xg_boost(100, COMBINED_DF, "cnn_xgb_graphs/predictions_plot.png", embedding_model,xgb_model)

history = xgb_model.evals_result()

graph_xgb_model_history(history, "cnn_xgb_graphs/xgb_model_history.png")
