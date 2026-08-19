import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from src.data_loader import load_and_preprocess_data
from src.scaling import scale_with_fitted, inverse_scale_y


def evaluate(target_crop):

    print("--- Loading Data & Saved Model ---")

    # Load selected crop data
    X_climate, X_soil, y = load_and_preprocess_data(
        target_crop=target_crop
    )

    # Split same as training
    _, X_clim_val, _, X_soil_val, _, y_val = train_test_split(
        X_climate,
        X_soil,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scale val inputs using the scalers fit during training (no fitting here)
    X_clim_val, X_soil_val, _ = scale_with_fitted(
        X_clim_val, X_soil_val, y_val,
        save_dir="saved_models", target_crop=target_crop
    )

    # Load matching crop model
    model_path = f"saved_models/best_{target_crop}_model.keras"

    print(f"Loading model: {model_path}")

    model = tf.keras.models.load_model(model_path)


    print("--- Running Predictions ---")

    y_pred = model.predict(
        {
            "Climate_Input": X_clim_val,
            "Soil_Input": X_soil_val
        },
        verbose=0
    ).flatten()

    # Bring predictions back out of scaled space into real mt/ha units
    y_pred = inverse_scale_y(y_pred, save_dir="saved_models", target_crop=target_crop)


    # Metrics (y_val is still raw/unscaled, y_pred is now unscaled too)
    r2 = r2_score(y_val, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_val, y_pred)
    )

    mae = mean_absolute_error(
        y_val,
        y_pred
    )

    print("\n==========================================")
    print("           MODEL EVALUATION METRICS       ")
    print("==========================================")
    print(f"Crop     : {target_crop}")
    print(f"R² Score : {r2:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"MAE      : {mae:.4f}")
    print("==========================================\n")


    # Actual vs Predicted Plot
    plt.figure(figsize=(7, 6))

    plt.scatter(
        y_val,
        y_pred,
        alpha=0.6,
        edgecolors="k"
    )

    min_val = min(min(y_val), min(y_pred))
    max_val = max(max(y_val), max(y_pred))

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        lw=2,
        label="Ideal Line (y=x)"
    )

    plt.xlabel("Actual Yield (mt/ha)")
    plt.ylabel("Predicted Yield (mt/ha)")

    plt.title(
        f"Actual vs Predicted {target_crop} Yield (R²={r2:.2f})"
    )

    plt.legend()
    plt.grid(True)


    # Same filename, overwrites previous graph
    plt.savefig(
        "saved_models/actual_vs_predicted.png"
    )

    plt.close()

    print(
        "Saved evaluation plot to: saved_models/actual_vs_predicted.png"
    )


if __name__ == "__main__":

    # Change only this crop name
    evaluate("WH_Y")