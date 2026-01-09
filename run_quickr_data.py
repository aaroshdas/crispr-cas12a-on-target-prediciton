import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from cnn_helper import convert_to_one_hot
file = './datasets/raw_quickr_seqs.csv'

df = pd.read_csv(file)
print(df.head())

#use cnn
weights_path = "./weights/saved_models/cnn/"
cnn_model_path = weights_path+ "save_01.keras"

seqs = df["Context Sequence"].values

def proccess_seqs(seqs):
    procssesed_seqs = []
    for s in seqs:
        new_s = s
        if len(s) > 34:
            new_s= s[:34]
        procssesed_seqs.append(new_s)
    return procssesed_seqs
procssesed_seqs = proccess_seqs(seqs)

for i in procssesed_seqs:
    print(i)

TARGET_MEAN = np.load("./weights/target_mean.npy")
TARGET_STD  = np.load("./weights/target_std.npy")

x_vals = np.array([convert_to_one_hot(seq) for seq in procssesed_seqs])

model = tf.keras.models.load_model(cnn_model_path)

y_pred_normalized = model.predict(x_vals).squeeze()
y_pred = (y_pred_normalized * TARGET_STD) + TARGET_MEAN
df["Indel frequency Prediction"] = y_pred
for i in y_pred:
    print(f'{i:.4f}')

print(df)
