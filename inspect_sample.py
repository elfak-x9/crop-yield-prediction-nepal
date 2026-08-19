import os

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from src.data_loader import load_and_preprocess_data


def inspect_sample(sample_idx=12):
    """Prints climate and soil features alongside the target yield for a specific sample index."""
    X_climate, X_soil, y = load_and_preprocess_data()

    print(f"\n==================================================")
    print(f"           INSPECTING SAMPLE #{sample_idx}")
    print(f"==================================================")
    print(f"Actual Yield: {y[sample_idx]:.3f} mt/ha\n")

    print("--- Soil & Tabular Features ---")
    print(f"Soil Features Vector: {X_soil[sample_idx]}\n")

    print("--- Climate Time-Series Summary ---")
    print(f"Climate Shape        : {X_climate[sample_idx].shape}")
    print(f"Climate Mean Values  : {np.mean(X_climate[sample_idx], axis=0)}")
    print(f"Climate Max Values   : {np.max(X_climate[sample_idx], axis=0)}")
    print(f"Climate Min Values   : {np.min(X_climate[sample_idx], axis=0)}")
    print(f"==================================================\n")
 # Find exact day and feature of the extreme spike
    day, feature_idx = np.unravel_index(np.argmax(X_climate[12]), X_climate[12].shape)
    print(f"Spike of {X_climate[12][day, feature_idx]:.2f} happened on Day {day} at Feature Index {feature_idx}")

if __name__ == "__main__":
    inspect_sample(12)