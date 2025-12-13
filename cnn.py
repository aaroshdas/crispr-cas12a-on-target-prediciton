#cnn 
#conda activate cc12on
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np



dataset_path = "./datasets/"
train_path = "Kim_2018_Train.csv"
test_path = "Kim_2018_Test.csv"


def filter_df(df):
    df = df.drop(columns=["50 bp synthetic target and target context sequence","20 bp guide sequence (5' to 3')","Indel frequency (% Background)","Indel read count (Background)","Total read count (Background)","Indel freqeuncy (Cpf1 delivered %)", "Indel read count (Cpf1 delivered)","Total read count (Cpf1 delivered)"], axis=1)
    df.rename(columns={"Context Sequence": "Input seq"}, inplace=True)
    df = df[df['Indel frequency'] >= -1]
    df['Indel frequency'] = df['Indel frequency'].clip(lower=0)
    
    #df['Indel frequency_norm'] = df['Indel frequency'] / 100.0

    return df

temp_train_df= filter_df(pd.read_csv(dataset_path + train_path))

temp_test_df= filter_df(pd.read_csv(dataset_path + test_path))

COMBINED_DF = pd.concat([temp_train_df,temp_test_df])

print(COMBINED_DF.head())

TARGET_MEAN = COMBINED_DF['Indel frequency'] .mean()
TARGET_STD = COMBINED_DF['Indel frequency'].std()

COMBINED_DF['Indel frequency'] = (COMBINED_DF['Indel frequency'] - TARGET_MEAN) / TARGET_STD

print(COMBINED_DF.head())
print("total samples", len(COMBINED_DF))


def convert_to_one_hot(seq):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    one_hot_temp_list= []
    for base in seq:
        one_hot_temp_list.append(mapping.get(base))
    return np.array(one_hot_temp_list)

# print(convert_to_one_hot(test_df.loc[0, "Input seq"]))

train_sequences = COMBINED_DF["Input seq"].values
raw_x_vals = np.array([convert_to_one_hot(seq) for seq in train_sequences])

raw_y_vals = COMBINED_DF["Indel frequency"].values.astype(float)


x_train, x_val, y_train, y_val = train_test_split(raw_x_vals, raw_y_vals, test_size=0.15)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)


model = models.Sequential([
    layers.Input(shape=(x_train.shape[1], 4)),
    layers.Conv1D(
            filters=128, 
            kernel_size=5, 
            activation='relu', 
            kernel_regularizer=regularizers.l2(0.0001)),
    # layers.BatchNormalization()
    layers.Dropout(0.2),

    layers.Conv1D(
            filters=256,
            kernel_size=3,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.0001)),
    
    # layers.BatchNormalization(),
    layers.GlobalMaxPooling1D(),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()





def make_prediction(model, seq):
    temp_batch_seq = convert_to_one_hot(seq)[np.newaxis, ...]
    pred = model.predict(temp_batch_seq, verbose=False)
    print(f'\n {pred * TARGET_STD + TARGET_MEAN}')

make_prediction(model, "AGCGTTTAAAAAACATCGAACGCATCTGCTGCCT") #14.711302
make_prediction(model, "AAACTTTAAAAATCTTTTCTGCCAGATCTCCAGA") #0.238095
make_prediction(model, "TTGTTTTAAAACAGGTTCTGTACTTGATCTCTCC") #88.079746


history = model.fit(x_train, y_train, epochs=10, batch_size =32, validation_data=(x_val, y_val))

model.save("cnn_model.keras")

make_prediction(model, "AGCGTTTAAAAAACATCGAACGCATCTGCTGCCT")
make_prediction(model, "AAACTTTAAAAATCTTTTCTGCCAGATCTCCAGA")
make_prediction(model, "TTGTTTTAAAACAGGTTCTGTACTTGATCTCTCC")

def graph_model_history(history):
    #loss vals
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])

    #mae vals
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model MAE')
    plt.ylabel('mAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.savefig("cnn_graphs/model_history.png")
    plt.show()

graph_model_history(history)

def plot_predictions(model, total_x_vals):
    temp_y_vals = []
    temp_pred_vals = []
    temp_x_vals = []
    for i in range(total_x_vals):
        temp_y_vals.append(COMBINED_DF.iloc[i, 1])
        temp_x_vals.append(i)
        seq = COMBINED_DF.iloc[i,  0]
        temp_batch_seq = convert_to_one_hot(seq)[np.newaxis, ...]

        pred = model.predict(temp_batch_seq, verbose=False)
        temp_pred_vals.append(pred[0][0])
    
    plt.figure(figsize=(12, 6))
    plt.plot(temp_x_vals, temp_y_vals, label='actual', linestyle='-', color='blue')
    plt.plot(temp_x_vals, temp_pred_vals, label='preds', linestyle='--', color='red')
    plt.ylabel('normalized indel freq')
    plt.savefig("cnn_graphs/predictions_plot.png")

plot_predictions(model, 300)