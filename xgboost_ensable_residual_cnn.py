import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def residual_block(filters, kernel_size=3, dilation_rate=1):
    def block(x):
        shortcut = x
        y = layers.Conv1D(
            filters,
            kernel_size,
            padding="same",
            dilation_rate=dilation_rate,
            kernel_regularizer=regularizers.l2(1e-4),
            use_bias=False
        )(x)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Conv1D(
            filters,
            kernel_size,
            padding="same",
            dilation_rate=dilation_rate,
            kernel_regularizer=regularizers.l2(1e-4),
            use_bias=False
        )(y)
        y = layers.BatchNormalization()(y)

        #residual connection
        y = layers.Add()([shortcut, y])
        y = layers.ReLU()(y)
        return y

    return block


def build_residual_dilated_cnn(seq_len=34):
    inp = layers.Input(shape=(seq_len, 4))

    x = layers.Conv1D(
        128, 5,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(inp)
    x = layers.BatchNormalization()(x)

    x = residual_block(128, kernel_size=3, dilation_rate=1)(x)
    x = residual_block(128, kernel_size=3, dilation_rate=2)(x)
    x = residual_block(128, kernel_size=3, dilation_rate=4)(x)

    x = layers.Dropout(0.3)(x)
    x = layers.GlobalMaxPooling1D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)

    model = models.Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"]
    )

    return model

def build_residual_cnn(x_train):
    seq_len = x_train.shape[1]
    model = build_residual_dilated_cnn(seq_len)
    model.summary()
    #model, pooling layer index
    return model, 12
