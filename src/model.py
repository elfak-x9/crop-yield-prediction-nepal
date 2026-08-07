import tensorflow as tf
from keras import layers, Model


def build_dual_input_cnn_lstm(
    time_steps, climate_features_count, soil_features_count
):
    """
    Dual-Branch Architecture with Conv1D + Bidirectional LSTM
    for sequence feature extraction and multi-stage Soil-Climate fusion.
    """
    # ---------------------------------------------------------
    # BRANCH 1: Climate Sequential Input (1D CNN + BiLSTM)
    # ---------------------------------------------------------
    climate_input = layers.Input(
        shape=(time_steps, climate_features_count), name="Climate_Input"
    )

    # 1D Convolution to capture local short-term daily patterns (e.g., 5-day weather spells)
    x_clim = layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu")(climate_input)
    x_clim = layers.BatchNormalization()(x_clim)
    x_clim = layers.MaxPooling1D(pool_size=2)(x_clim)

    # Second CNN block for deeper temporal pattern extraction
    x_clim = layers.Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(x_clim)
    x_clim = layers.BatchNormalization()(x_clim)
    x_clim = layers.MaxPooling1D(pool_size=2)(x_clim)

    # Bidirectional LSTM to capture seasonal trends forward and backward
    x_clim = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x_clim)
    x_clim = layers.BatchNormalization()(x_clim)
    x_clim = layers.Dropout(0.2)(x_clim)

    # ---------------------------------------------------------
    # BRANCH 2: Soil Input (Deep Tabular Dense Branch)
    # ---------------------------------------------------------
    soil_input = layers.Input(
        shape=(soil_features_count,), name="Soil_Input"
    )

    x_soil = layers.Dense(64, activation="relu")(soil_input)
    x_soil = layers.BatchNormalization()(x_soil)
    x_soil = layers.Dense(32, activation="relu")(x_soil)
    x_soil = layers.Dropout(0.2)(x_soil)

    # ---------------------------------------------------------
    # FUSION & DENSE REGRESSION HEAD
    # ---------------------------------------------------------
    combined = layers.concatenate([x_clim, x_soil], name="Feature_Fusion")

    x = layers.Dense(128, activation="relu")(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(32, activation="relu")(x)

    # Linear output for target yield (mt/ha)
    output = layers.Dense(1, activation="linear", name="Yield_Output")(x)

    # Compile model with Adam optimizer and Huber Loss
    model = Model(inputs=[climate_input, soil_input], outputs=output)
    
   # Change this line in model.py:
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0), # Added clipnorm
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=["mean_absolute_error", "mean_squared_error"]
)

    return model