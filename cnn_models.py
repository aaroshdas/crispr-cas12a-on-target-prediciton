from tensorflow.keras import layers, models, regularizers # type: ignore
import xgboost_ensemble_cnn_models
import tensorflow as tf
def load_standard_model(x_train):
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
    return model


def load_residual_model(x_train):
        #load the xgboost version
        model = xgboost_ensemble_cnn_models.build_residual_cnn(x_train)
        return model





def residual_block(filters, kernel_size=3, dilation_rate=1, dropout_rate=0.15):
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
        y = layers.SpatialDropout1D(dropout_rate)(y)

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


def load_multitask_model(x_train):
        seq_len= x_train.shape[1]
        
        inp = layers.Input(shape=(seq_len, 4))

        x = layers.Conv1D(64, 5,padding="same",activation="relu",kernel_regularizer=regularizers.l2(2e-4))(inp)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = residual_block(64, kernel_size=3, dilation_rate=1)(x)
        x = residual_block(64, kernel_size=3, dilation_rate=2)(x)
        x = residual_block(64, kernel_size=3, dilation_rate=4)(x)
        x = layers.Dropout(0.3)(x)

        max_pool = layers.GlobalMaxPooling1D()(x)
        avg_pool = layers.GlobalAveragePooling1D()(x)
        
        z = layers.Concatenate(name="embedding_2")([max_pool, avg_pool])

        z = layers.Dense(96,activation="relu",kernel_regularizer=regularizers.l2(1e-4), name="embedding")(x)
        z = layers.Dropout(0.4)(x)

        y_regression = layers.Dense(1, activation="linear", name="regression")(z)
        y_binary = layers.Dense(1, activation=None, name="binary")(z)

        model = models.Model(inputs=inp, outputs={
            "regression": y_regression,
            "binary": y_binary
        })

        model.compile(
                optimizer=tf.keras.optimizers.Adam(3e-4),
                loss={
                        "regression": tf.keras.losses.Huber(delta=1.0),
                        "binary": tf.keras.losses.BinaryCrossentropy(from_logits=True),
                },
                loss_weights={
                        "regression": 1.0,
                        "binary": 0.3,
                },
                metrics={
                        "regression": ["mae"],
                        "binary": ["accuracy"]
                }
        )

        return model

      