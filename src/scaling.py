import joblib
from sklearn.preprocessing import StandardScaler


def fit_and_scale_train(X_clim_train, X_soil_train, y_train, save_dir, target_crop):
    """
    Fits scalers on TRAIN data only, transforms train arrays, and
    saves the fitted scalers so evaluate.py can reuse them.
    """
    n_samples, seq_len, n_climate_feats = X_clim_train.shape

    # Climate is 3D (samples, timesteps, features) -> flatten to 2D to fit
    scaler_climate = StandardScaler()
    clim_flat = X_clim_train.reshape(-1, n_climate_feats)
    clim_flat_scaled = scaler_climate.fit_transform(clim_flat)
    X_clim_train_scaled = clim_flat_scaled.reshape(n_samples, seq_len, n_climate_feats)

    # Soil is already 2D
    scaler_soil = StandardScaler()
    X_soil_train_scaled = scaler_soil.fit_transform(X_soil_train)

    # Target
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    # Persist scalers for evaluate.py
    joblib.dump(scaler_climate, f"{save_dir}/{target_crop}_scaler_climate.pkl")
    joblib.dump(scaler_soil, f"{save_dir}/{target_crop}_scaler_soil.pkl")
    joblib.dump(scaler_y, f"{save_dir}/{target_crop}_scaler_y.pkl")

    return X_clim_train_scaled, X_soil_train_scaled, y_train_scaled


def scale_with_fitted(X_clim, X_soil, y, save_dir, target_crop):
    """
    Loads previously-fit scalers and transforms val/eval data with them
    (never fit again here).
    """
    scaler_climate = joblib.load(f"{save_dir}/{target_crop}_scaler_climate.pkl")
    scaler_soil = joblib.load(f"{save_dir}/{target_crop}_scaler_soil.pkl")
    scaler_y = joblib.load(f"{save_dir}/{target_crop}_scaler_y.pkl")

    n_samples, seq_len, n_climate_feats = X_clim.shape
    clim_flat = X_clim.reshape(-1, n_climate_feats)
    clim_flat_scaled = scaler_climate.transform(clim_flat)
    X_clim_scaled = clim_flat_scaled.reshape(n_samples, seq_len, n_climate_feats)

    X_soil_scaled = scaler_soil.transform(X_soil)

    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten() if y is not None else None

    return X_clim_scaled, X_soil_scaled, y_scaled


def inverse_scale_y(y_scaled, save_dir, target_crop):
    scaler_y = joblib.load(f"{save_dir}/{target_crop}_scaler_y.pkl")
    return scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()