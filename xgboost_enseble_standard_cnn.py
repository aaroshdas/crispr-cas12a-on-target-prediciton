import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_standard_cnn(x_train):
    model = models.Sequential([
        layers.Input(shape=(x_train.shape[1], 4)),
        layers.Conv1D(
                filters=128, 
                kernel_size=5, 
                activation='relu', 
                kernel_regularizer=regularizers.l2(0.0001)),
        layers.Dropout(0.2),

        layers.Conv1D(
                filters=256,
                kernel_size=3,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.0001)),
        
        # layers.Dropout(0.2),
        # layers.Conv1D(
        #     filters=256,
        #     kernel_size=2,
        #     activation='relu',
        #     kernel_regularizer=regularizers.l2(0.0001)),

        layers.GlobalMaxPooling1D(),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(1, activation='linear')
    ])
    #model, pooling layer index
    return model, 4