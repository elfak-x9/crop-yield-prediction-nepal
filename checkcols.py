import pandas as pd


soil_path = "data/processed/nepal_75_districts_soil_dataset.csv"
climate_path = "data/processed/nepal_district_climate_1979_2024.csv"
yield_path= "data/processed/crops-yield-1979-2024.csv"

print("--- SOIL FILE COLUMNS ---")
try:
    df_soil = pd.read_csv(soil_path)
    print(df_soil.columns.tolist())
except Exception as e:
    print(e)

print("\n--- CLIMATE FILE COLUMNS ---")
try:
    df_climate = pd.read_csv(climate_path)
    print(df_climate.columns.tolist())
except Exception as e:
    print(e)

print("\n--- YIELD FILE COLUMNS ---")
try:
    df_yield = pd.read_csv(yield_path)
    print(df_yield.columns.tolist())
except Exception as e:
    print(e)