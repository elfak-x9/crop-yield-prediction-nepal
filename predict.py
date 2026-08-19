import tensorflow as tf

from src.data_loader import load_and_preprocess_data
from src.scaling import scale_with_fitted, inverse_scale_y


def predict_samples(
    target_crop,
    selected_districts,
    selected_years,
    save_dir="saved_models",
):

    print("\n==============================")
    print(f"Predicting {target_crop}")
    print("==============================")

    # --------------------------------------------------
    # Check districts and years
    # --------------------------------------------------
    if len(selected_districts) != len(selected_years):
        raise ValueError(
            "selected_districts and selected_years must have "
            "the same number of items."
        )

    # --------------------------------------------------
    # Load correct model
    # --------------------------------------------------
    model_path = f"{save_dir}/best_{target_crop}_model.keras"

    print(f"Loading model: {model_path}")

    model = tf.keras.models.load_model(model_path)

    # --------------------------------------------------
    # Load crop data + metadata
    # --------------------------------------------------
    X_climate, X_soil, y, districts, years = load_and_preprocess_data(
        target_crop=target_crop,
        sequence_length=12,
        return_metadata=True,
    )

    # --------------------------------------------------
    # Find exact district + year samples
    # --------------------------------------------------
    selected_indices = []

    for requested_district, requested_year in zip(
        selected_districts,
        selected_years,
    ):

        requested_district = (
            requested_district.strip().lower()
        )

        requested_year = int(requested_year)

        matching_indices = [
            i
            for i, (district, year) in enumerate(
                zip(districts, years)
            )
            if district == requested_district
            and int(year) == requested_year
        ]

        if len(matching_indices) == 0:

            print(
                f"WARNING: No valid sample found for "
                f"{requested_district} in {requested_year}."
            )

        else:

            selected_indices.append(
                matching_indices[0]
            )

    # --------------------------------------------------
    # Stop if no samples were found
    # --------------------------------------------------
    if len(selected_indices) == 0:
        print("No valid district/year combinations found.")
        return

    # --------------------------------------------------
    # Select requested samples
    # --------------------------------------------------
    selected_climate = X_climate[selected_indices]
    selected_soil = X_soil[selected_indices]
    selected_actual = y[selected_indices]

    selected_district_names = [
        districts[i]
        for i in selected_indices
    ]

    selected_year_values = [
        years[i]
        for i in selected_indices
    ]

    # --------------------------------------------------
    # Scale using the SAME scalers used during training
    # --------------------------------------------------
    selected_climate, selected_soil, _ = scale_with_fitted(
        selected_climate,
        selected_soil,
        None,
        save_dir=save_dir,
        target_crop=target_crop,
    )

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------
    predictions = model.predict(
        {
            "Climate_Input": selected_climate,
            "Soil_Input": selected_soil,
        },
        verbose=0,
    ).flatten()

    # --------------------------------------------------
    # Convert predictions back to mt/ha
    # --------------------------------------------------
    predictions = inverse_scale_y(
        predictions,
        save_dir=save_dir,
        target_crop=target_crop,
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------
    print(
        "\n=========================================================================="
    )

    print(
        "District       | Year | Actual (mt/ha) | "
        "Predicted (mt/ha) | Diff"
    )

    print(
        "=========================================================================="
    )

    for (
        district,
        year,
        actual,
        predicted,
    ) in zip(
        selected_district_names,
        selected_year_values,
        selected_actual,
        predictions,
    ):

        diff = abs(actual - predicted)

        print(
            f"{district:<14} | "
            f"{year:<4} | "
            f"{actual:<15.3f} | "
            f"{predicted:<17.3f} | "
            f"{diff:.3f}"
        )

    print(
        "=========================================================================="
    )


if __name__ == "__main__":

    predict_samples(
        target_crop="WH_Y",

        selected_districts=[
            "arghakhanchi",
            "ilam",
            "morang",
            "jhapa",
            "khotang",
            "sindhuli",
        ],

        selected_years=[
            1981,
            1990,
            2000,
            2010,
            2020,
            2024,
        ],
    )