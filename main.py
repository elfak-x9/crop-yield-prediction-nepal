import os
import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from src.data_loader import load_and_preprocess_data
from src.model import build_dual_input_cnn_lstm
from src.train import train_and_evaluate_model
from src.scaling import fit_and_scale_train, scale_with_fitted


# ==================================================
# Configuration
# ==================================================
SEED = 42
TARGET_CROP = "WH_Y"       # Change to PD_Y, MZ_Y, etc.
SEQUENCE_LENGTH = 150
EPOCHS = 100
BATCH_SIZE = 16
SAVE_DIR = "saved_models"


# ==================================================
# Set random seed for reproducibility
# ==================================================
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def main():

    soil_csv = "data/processed/nepal_75_districts_soil_dataset.csv"
    climate_csv = "data/processed/nepal_district_climate_1979_2024.csv"
    yield_csv = "data/processed/crops-yield-1979-2024.csv"

    print("==================================================")
    print("Step 1: Loading & Preprocessing Data...")
    print("==================================================")

    X_climate, X_soil, y = load_and_preprocess_data(
        soil_path=soil_csv,
        climate_path=climate_csv,
        yield_path=yield_csv,
        target_crop=TARGET_CROP,
        sequence_length=SEQUENCE_LENGTH,
    )

    print(f"Climate Input Shape (X_climate) : {X_climate.shape}")
    print(f"Soil Input Shape    (X_soil)    : {X_soil.shape}")
    print(f"Yield Target Shape  (y)         : {y.shape}")

    if len(y) == 0:
        raise ValueError("No matching samples found!")

    print("\nSplitting Dataset (80% Train, 20% Validation)...")

    (
        X_clim_train,
        X_clim_val,
        X_soil_train,
        X_soil_val,
        y_train,
        y_val,
    ) = train_test_split(
        X_climate,
        X_soil,
        y,
        test_size=0.2,             
        random_state=SEED,
    )

    print("\nScaling Features & Target (fit on TRAIN only)...")

    os.makedirs(SAVE_DIR, exist_ok=True)

    X_clim_train, X_soil_train, y_train = fit_and_scale_train(
        X_clim_train, X_soil_train, y_train,
        save_dir=SAVE_DIR, target_crop=TARGET_CROP,
    )

    X_clim_val, X_soil_val, y_val_scaled = scale_with_fitted(
        X_clim_val, X_soil_val, y_val,
        save_dir=SAVE_DIR, target_crop=TARGET_CROP,
    )

    # Keep the raw y_val around for later human-readable inspection if needed;
    # train_and_evaluate_model needs the SCALED version to match y_train's scale.
    y_val = y_val_scaled


    print("\n==================================================")
    print("Step 2: Building Dual-Branch 1D CNN-LSTM Model...")
    print("==================================================")

    model = build_dual_input_cnn_lstm(
        time_steps=X_climate.shape[1],
        climate_features_count=X_climate.shape[2],
        soil_features_count=X_soil.shape[1],
    )

    model.summary()


    print("\n==================================================")
    print("Step 3: Training Model...")
    print("==================================================")

    history = train_and_evaluate_model(
        model=model,
        X_clim_train=X_clim_train,
        X_clim_val=X_clim_val,
        X_soil_train=X_soil_train,
        X_soil_val=X_soil_val,
        y_train=y_train,
        y_val=y_val,
        target_crop=TARGET_CROP,   # Saves best_ML_Y_model.keras
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        save_dir=SAVE_DIR,
    )


    print("\n================ BEST TRAINING METRICS ================")

    mae_key = "mae" if "mae" in history.history else "mean_absolute_error"
    val_mae_key = (
    "val_mae"
    if "val_mae" in history.history
    else "val_mean_absolute_error"
    )

    # Find the epoch with the lowest validation loss
    best_epoch = np.argmin(history.history["val_loss"])

    print(f"Best Epoch            : {best_epoch + 1}")
    print(f"Best Training Loss    : {history.history['loss'][best_epoch]:.6f}")
    print(f"Best Validation Loss  : {history.history['val_loss'][best_epoch]:.6f}")
    print(f"Best Training MAE     : {history.history[mae_key][best_epoch]:.6f}")
    print(f"Best Validation MAE   : {history.history[val_mae_key][best_epoch]:.6f}")

print("(Loss/MAE above are in SCALED units.)")
print("Real mt/ha performance is reported by evaluate.py.")

print("========================================================")


if __name__ == "__main__":
    main()