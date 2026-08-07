import pandas as pd
from src.preprocessing import process_features_and_build_sequences


def load_and_preprocess_data(
    soil_path="data/processed/nepal_75_districts_soil_dataset.csv",
    climate_path="data/processed/nepal_district_climate_1979_2024.csv",
    yield_path="data/processed/crops-yield-1979-2024.csv",
    target_crop="PD_Y",
    sequence_length=150,
):
    # 1. Load Datasets
    soil_df = pd.read_csv(soil_path)
    climate_df = pd.read_csv(climate_path)
    yield_df = pd.read_csv(yield_path)

    # Strip whitespace from columns
    soil_df.columns = soil_df.columns.str.strip()
    climate_df.columns = climate_df.columns.str.strip()
    yield_df.columns = yield_df.columns.str.strip()

    # Pass loaded dataframes to preprocessing step
    return process_features_and_build_sequences(
        soil_df=soil_df,
        climate_df=climate_df,
        yield_df=yield_df,
        target_crop=target_crop,
        sequence_length=sequence_length,
    )