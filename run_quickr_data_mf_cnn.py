import pandas as pd
import numpy as np
import tensorflow as tf
# from tensorflow.keras import models
from cnn_helper import convert_to_one_hot

file = './quickr_data/NF_raw_quickr_seqs.csv'

df = pd.read_csv(file)
print(df.head())

#use cnn
weights_path = "./weights/saved_models/multi_feature/"
cnn_model_path = weights_path+ "weights.keras"

seqs = df["Context Sequence"].values

x_seq = np.array([convert_to_one_hot(seq) for seq in seqs])

x_feat = df[["gc content", "pam_prox_at_frac", "pos18_is_c"]].values.astype(np.float32)


TARGET_MEAN = np.load("./weights/target_mean.npy")
TARGET_STD  = np.load("./weights/target_std.npy")

model = tf.keras.models.load_model(cnn_model_path)

y_pred_normalized = model.predict([x_seq, x_feat], batch_size=64).squeeze()
y_pred = (y_pred_normalized * TARGET_STD) + TARGET_MEAN

df["Indel frequency Prediction"] = y_pred

for i in y_pred:
    print(f'{i:.4f}')
print(df)
