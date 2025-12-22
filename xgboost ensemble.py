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


dataset_path = "./datasets/"
train_path = "Kim_2018_Train.csv"
test_path = "Kim_2018_Test.csv"

#csv needs to be formated: Context Sequence, Indel frequency


temp_train_df= filter_df(pd.read_csv(dataset_path + train_path))

temp_test_df= filter_df(pd.read_csv(dataset_path + test_path))

TEST_DF = temp_test_df.iloc[-100:]
temp_test_df = temp_test_df.iloc[:-100]



COMBINED_DF = pd.concat([temp_train_df,temp_test_df])


print(COMBINED_DF.head())

# TARGET_MEAN = COMBINED_DF['Indel frequency'] .mean()
# TARGET_STD = COMBINED_DF['Indel frequency'].std()

TARGET_MEAN = np.load("./weights/target_mean.npy")
TARGET_STD  = np.load("./weights/target_std.npy")

COMBINED_DF['Indel frequency'] = (COMBINED_DF['Indel frequency'] - TARGET_MEAN) / TARGET_STD

TEST_DF = COMBINED_DF.iloc[-100:]
COMBINED_DF = COMBINED_DF.iloc[:-100]


train_sequences = COMBINED_DF["Input seq"].values

raw_x_vals = np.array([convert_to_one_hot(seq, ) for seq in train_sequences])
raw_y_vals = COMBINED_DF["Indel frequency"].values.astype(float)


x_train, x_val, y_train, y_val = train_test_split(raw_x_vals, raw_y_vals, test_size=0.15)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)



#original 4#
MAX_POOLING_LAYER_INDEX = 4

#RESIDUAL CNN
model,MAX_POOLING_LAYER_INDEX = xgboost_ensemble_cnn_models.build_residual_cnn(x_train)

#model,MAX_POOLING_LAYER_INDEX= xgboost_ensemble_cnn_models.build_standard_cnn(x_train)



model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()


history = model.fit(x_train, y_train, epochs=12, batch_size=32, validation_data=(x_val, y_val))

graph_model_history(history, "cnn_xgb_graphs/cnn_feature_extractor_history.png", "mae")



model.build(input_shape=(None, x_train.shape[1], 4))

embedding_model = models.Model(inputs=model.inputs, outputs=model.layers[MAX_POOLING_LAYER_INDEX].output)

model.save("./weights/cnn_embeddings_model.keras")


x_train_embed = embedding_model(x_train, training=False).numpy()
x_val_embed = embedding_model(x_val, training=False).numpy()

print("embedding shape", x_train_embed.shape)

tnse_embedding_visualization(x_train_embed, y_train)

umap_embedding_visualization(x_train_embed, y_train)



xgb_model = xgb.XGBRegressor(
    n_estimators=250,
    learning_rate=0.03,
    max_depth=5,

    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0
    )


xgb_model.fit(x_train_embed, y_train,
              eval_set=[(x_val_embed, y_val)],
              verbose=True)

xgb_model.save_model("./weights/cnn_xgb_model.json")



plot_predictions_xg_boost(len(TEST_DF), TEST_DF, "cnn_xgb_graphs/predictions_plot.png", embedding_model,xgb_model)
#plot_predictions_xg_boost(100, COMBINED_DF, "cnn_xgb_graphs/predictions_plot.png", embedding_model,xgb_model)

history = xgb_model.evals_result()

graph_xgb_model_history(history, "cnn_xgb_graphs/xgb_model_history.png")
