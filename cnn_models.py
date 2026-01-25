from tensorflow.keras import layers, models, regularizers # type: ignore
import xgboost_ensemble_cnn_models
import tensorflow as tf

from xgboost_ensemble_cnn_models import residual_block

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



def build_residual_multi_feature_cnn(seq_len=34):
        seq_inp = layers.Input(shape=(seq_len, 4), name="seq_input")  
        feat_inp = layers.Input(shape=(3,), name="feat_input")


        x = layers.Conv1D(64, 5,padding="same",activation="relu",kernel_regularizer=regularizers.l2(2e-4))(seq_inp)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = residual_block(64, kernel_size=3, dilation_rate=1)(x)
        x = residual_block(64, kernel_size=3, dilation_rate=2)(x)
        x = residual_block(64, kernel_size=3, dilation_rate=4)(x)
        x = layers.Dropout(0.3)(x)

        max_pool = layers.GlobalMaxPooling1D()(x)
        avg_pool = layers.GlobalAveragePooling1D()(x)

        seq_embed = layers.Concatenate(name="embedding_2")([max_pool, avg_pool])
        seq_embed = layers.Dense(96, activation="relu", kernel_regularizer=regularizers.l2(1e-4),name="embedding")(seq_embed)

        feat_x = layers.Dense(16, activation="relu")(feat_inp)
        feat_x = layers.Dense(16, activation="relu")(feat_x)

        x = layers.Concatenate()([seq_embed, feat_x])
        
        x = layers.Dropout(0.4)(x)
        out = layers.Dense(1, activation="linear")(x)


        model = models.Model(inputs=[seq_inp, feat_inp], outputs=out)

        model.compile(
                optimizer=tf.keras.optimizers.Adam(3e-4),
                loss="mse",
                metrics=["mae"]
        )
        # model.compile(
        #     optimizer=tf.keras.optimizers.Adam(2e-4),
        #     loss=tf.keras.losses.Huber(delta=1.0),
        #     metrics=["mae"]
        # )

        return model

