from tensorflow.keras import layers, models, regularizers
import xgboost_ensemble_cnn_models
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
        model, _ = xgboost_ensemble_cnn_models.build_residual_cnn(x_train)
        return model