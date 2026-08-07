import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def train_and_evaluate_model(
    model,
    X_clim_train,
    X_clim_val,
    X_soil_train,
    X_soil_val,
    y_train,
    y_val,
    target_crop,
    epochs=100,
    batch_size=16,
    save_dir="saved_models",
):
    """
    Trains the dual-input CNN-LSTM model,
    saves the best model for the selected crop,
    and plots training history to training_history.png.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Saved model target file (e.g., saved_models/best_ML_Y_model.keras)
    model_save_path = os.path.join(
        save_dir,
        f"best_{target_crop}_model.keras"
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Flexible Input Selection (Positional List vs. Named Dict)
    try:
        train_inputs = [X_clim_train, X_soil_train]
        val_inputs = [X_clim_val, X_soil_val]
    except Exception:
        train_inputs = {
            "Climate_Input": X_clim_train,
            "Soil_Input": X_soil_train,
        }
        val_inputs = {
            "Climate_Input": X_clim_val,
            "Soil_Input": X_soil_val,
        }

    history = model.fit(
        x=train_inputs,
        y=y_train,
        validation_data=(val_inputs, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print(
        f"\nTraining Complete!"
        f"\nBest model saved to: {model_save_path}"
    )

    # Overwrites the fixed single training_history.png file
    plot_training_curves(
        history,
        os.path.join(save_dir, "training_history.png")
    )

    return history


def plot_training_curves(history, save_path):
    """
    Generates and saves Loss and MAE performance curves to training_history.png.
    """

    plt.figure(figsize=(12, 5))

    # -----------------------------
    # Loss Plot
    # -----------------------------
    plt.subplot(1, 2, 1)

    plt.plot(
        history.history["loss"],
        label="Train Loss",
        color="#1f77b4",
        lw=2
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
        color="#ff7f0e",
        lw=2
    )

    plt.title("Model Loss Curves", fontweight="bold")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # -----------------------------
    # MAE Plot
    # -----------------------------
    mae_key = (
        "mae"
        if "mae" in history.history
        else "mean_absolute_error"
    )

    val_mae_key = (
        "val_mae"
        if "val_mae" in history.history
        else "val_mean_absolute_error"
    )

    plt.subplot(1, 2, 2)

    if mae_key in history.history and val_mae_key in history.history:
        plt.plot(
            history.history[mae_key],
            label="Train MAE",
            color="#2ca02c",
            lw=2
        )

        plt.plot(
            history.history[val_mae_key],
            label="Validation MAE",
            color="#d62728",
            lw=2
        )

        plt.title("Model Metric (MAE)", fontweight="bold")
        plt.xlabel("Epochs")
        plt.ylabel("MAE (mt/ha)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    print(
        f"Training loss curve saved to: {save_path}"
    )

    plt.close()