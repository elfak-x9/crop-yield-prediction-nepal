import numpy as np
import pandas as pd


def process_features_and_build_sequences(
    soil_df, climate_df, yield_df, target_crop, sequence_length=150
):
    # 1. Parse Dates and Standardize District Names
    climate_df["DATE"] = pd.to_datetime(climate_df["DATE"])
    climate_df["Year"] = climate_df["DATE"].dt.year.astype(int)

    yield_df["Year"] = (
        yield_df["Date"]
        .astype(str)
        .str.split("-")
        .str[0]
        .str.strip()
        .astype(int)
    )

    # Standardize District names (Lowercase + Strip Whitespace)
    soil_df["District"] = soil_df["District"].astype(str).str.strip().str.lower()
    yield_df["District"] = yield_df["District"].astype(str).str.strip().str.lower()

    # 2. Map Soil Districts to Nearest Climate Coordinates
    climate_coords = climate_df[["LAT", "LON"]].drop_duplicates().values

    district_to_coord = {}
    for _, row in soil_df.iterrows():
        dist = row["District"]
        s_lat, s_lon = float(row["lat"]), float(row["lon"])

        # Calculate Euclidean distance to find closest grid point
        distances = np.sqrt(
            (climate_coords[:, 0] - s_lat) ** 2
            + (climate_coords[:, 1] - s_lon) ** 2
        )
        closest_idx = np.argmin(distances)
        district_to_coord[dist] = (
            climate_coords[closest_idx][0],
            climate_coords[closest_idx][1],
        )

    # 3. Process Soil Features (RAW - no scaling here; scaling happens
    #    after train/val split, in src/scaling.py)
    soil_cols = [
        "ph",
        "organic_matter_pct",
        "total_nitrogen_pct",
        "p2o5_kgha",
        "potassium_kgha",
        "zinc_ppm",
        "boron_ppm",
        "sand_pct",
        "clay_pct",
        "silt_pct",
    ]

    soil_scaled = soil_df[soil_cols].values

    # One-hot encode parent_soil
    soil_dummies = pd.get_dummies(
        soil_df["parent_soil"], prefix="soil_type", dtype=float
    ).values

    X_soil_dict = {}
    for idx, row in soil_df.iterrows():
        district = row["District"]
        combined = np.hstack([soil_scaled[idx], soil_dummies[idx]])
        X_soil_dict[district] = combined

    # 4. Weather Features (RAW - no scaling here; scaling happens
    #    after train/val split, in src/scaling.py)
    climate_features = ["PRECTOT", "RH2M", "T2M", "T2M_MAX", "T2M_MIN", "WS10M"]

    # 5. Build Sequences
    X_climate_list, X_soil_list, y_yield_list = [], [], []

    for _, row in yield_df.iterrows():
        district = row["District"]
        year = int(row["Year"])
        target_yield = float(row[target_crop]) if pd.notna(row[target_crop]) else np.nan

        # FILTER BAD/ZERO YIELDS: Skip missing, NaN, or non-positive yields
        if district not in X_soil_dict or pd.isna(target_yield) or target_yield <= 0.0:
            continue

        if district not in district_to_coord:
            continue

        c_lat, c_lon = district_to_coord[district]

        # Match weather data for district's coordinate grid point and target year
        sub_climate = climate_df[
            np.isclose(climate_df["LAT"], c_lat)
            & np.isclose(climate_df["LON"], c_lon)
            & (climate_df["Year"] == year)
        ].sort_values("DATE")

        if len(sub_climate) > 0:
            seq = sub_climate[climate_features].values

            # Slice or pad to exact sequence length
            if len(seq) >= sequence_length:
                seq = seq[:sequence_length]
            else:
                # Edge pad (repeat last row) instead of zero padding for weather consistency
                pad_width = sequence_length - len(seq)
                padding = np.tile(seq[-1:], (pad_width, 1))
                seq = np.vstack([seq, padding])

            X_climate_list.append(seq)
            X_soil_list.append(X_soil_dict[district])
            y_yield_list.append(target_yield)

    return (
        np.array(X_climate_list, dtype=np.float32),
        np.array(X_soil_list, dtype=np.float32),
        np.array(y_yield_list, dtype=np.float32),
    )