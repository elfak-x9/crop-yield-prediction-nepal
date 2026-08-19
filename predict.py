import numpy as np
import tensorflow as tf

from src.data_loader import load_and_preprocess_data
from src.scaling import scale_with_fitted, inverse_scale_y


def predict_samples(target_crop, indices, save_dir="saved_models"):

    print("\n==============================")
    print(f"Predicting {target_crop}")
    print("==============================")


    # Load correct model automatically
    model_path = f"{save_dir}/best_{target_crop}_model.keras"

    print(f"Loading model: {model_path}")

    model = tf.keras.models.load_model(
        model_path
    )


    # Load correct crop data (raw)
    X_climate, X_soil, y = load_and_preprocess_data(
        target_crop=target_crop
    )


    # Select requested samples (raw)
    selected_climate = X_climate[indices]
    selected_soil = X_soil[indices]
    selected_actual = y[indices]

    # Scale inputs using the SAME scalers fit during training
    selected_climate, selected_soil, _ = scale_with_fitted(
        selected_climate, selected_soil, None,
        save_dir=save_dir, target_crop=target_crop
    )

    # Predict all samples together
    predictions = model.predict(
        {
            "Climate_Input": selected_climate,
            "Soil_Input": selected_soil
        },
        verbose=0
    ).flatten()

    # Bring predictions back into real mt/ha units
    predictions = inverse_scale_y(predictions, save_dir=save_dir, target_crop=target_crop)


    # Print table
    print("\n==============================================================")
    print("Index    | Actual (mt/ha)  | Predicted (mt/ha)  | Diff")
    print("==============================================================")

    for idx, actual, predicted in zip(indices, selected_actual, predictions):

        diff = abs(actual - predicted)

        print(
            f"{idx:<9}| "
            f"{actual:<16.3f}| "
            f"{predicted:<19.3f}| "
            f"{diff:.3f}"
        )

    print("==============================================================\n")



if __name__ == "__main__":

    predict_samples(
        target_crop="BW_Y",
        indices=[5, 12, 25, 42, 50]
    )