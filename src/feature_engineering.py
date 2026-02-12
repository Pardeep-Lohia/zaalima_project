import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path

from .utils import setup_logging, ensure_directory

logger = setup_logging()

class FeatureEngineer:
    """
    Production-ready feature engineering pipeline for IoT predictive maintenance.
    """

    def __init__(self, sensor_columns: List[str] = None):
        """
        Initialize the feature engineer.

        Args:
            sensor_columns: List of sensor column names to engineer features for
        """
        if sensor_columns is None:
            sensor_columns = ['temperature', 'vibration', 'pressure']
        self.sensor_columns = sensor_columns
        self.scaler = StandardScaler()
        self.is_fitted = False

    def create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced time-series features for each sensor.

        Args:
            df: Input DataFrame with timestamp and sensor columns

        Returns:
            DataFrame with additional engineered features
        """
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Convert timestamp to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set timestamp as index for rolling operations
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()

        logger.info("Creating time-series features...")

        for sensor in self.sensor_columns:
            logger.info(f"Engineering features for {sensor}")

            # Rolling statistics with different windows
            windows = ['1h', '6h', '12h']

            for window in windows:
                df[f'{sensor}_rolling_mean_{window}'] = (
                    df[sensor].rolling(window=window, min_periods=1).mean()
                )

                df[f'{sensor}_rolling_std_{window}'] = (
                    df[sensor].rolling(window=window, min_periods=2).std()
                )

                df[f'{sensor}_rolling_var_{window}'] = (
                    df[sensor].rolling(window=window, min_periods=2).var()
                )

                # Rolling max and min
                df[f'{sensor}_rolling_max_{window}'] = (
                    df[sensor].rolling(window=window, min_periods=1).max()
                )

                df[f'{sensor}_rolling_min_{window}'] = (
                    df[sensor].rolling(window=window, min_periods=1).min()
                )

                # Rolling range (max - min)
                df[f'{sensor}_rolling_range_{window}'] = (
                    df[f'{sensor}_rolling_max_{window}'] - df[f'{sensor}_rolling_min_{window}']
                )


            # Exponential moving averages
            df[f'{sensor}_ema_12'] = df[sensor].ewm(span=12, adjust=False).mean()
            df[f'{sensor}_ema_24'] = df[sensor].ewm(span=24, adjust=False).mean()

            # Lag features
            df[f'{sensor}_lag_1'] = df[sensor].shift(1)
            df[f'{sensor}_lag_2'] = df[sensor].shift(2)

            # Rate of change (difference from lag)
            df[f'{sensor}_roc_1'] = df[sensor] - df[f'{sensor}_lag_1']

            # Rolling slope (trend over last 6h using linear regression)
            df[f'{sensor}_rolling_slope_6h'] = self._calculate_rolling_slope(df[sensor], '6h')

        # Reset index
        df.reset_index(inplace=True)

        # Add interaction features
        self._add_interaction_features(df)

        # Add delta features
        self._add_delta_features(df)

        # Add failure proximity features
        self._add_failure_proximity_features(df)

        # Handle NaNs created by rolling/lag operations
        df = self._handle_nans(df)

        logger.info(f"Created {len(df.columns) - 4} additional features")  # -4 for timestamp and original sensors + failure

        return df

    def _calculate_rolling_slope(self, series: pd.Series, window: str) -> pd.Series:
        """
        Calculate rolling slope using linear regression over a time window.

        Args:
            series: Time series data
            window: Rolling window size (e.g., '6h')

        Returns:
            Series with rolling slope values
        """
        def slope_func(x):
            if len(x) < 2:
                return 0
            # Create time index for regression
            t = np.arange(len(x))
            try:
                slope = np.polyfit(t, x, 1)[0]
                return slope
            except:
                return 0

        return series.rolling(window=window, min_periods=2).apply(slope_func, raw=False)

    def _add_interaction_features(self, df: pd.DataFrame) -> None:
        """
        Add interaction features between sensors.

        Args:
            df: DataFrame with sensor columns
        """
        # Basic interactions
        df['temperature_vibration'] = df['temperature'] * df['vibration']
        df['temperature_pressure'] = df['temperature'] / (df['pressure'] + 1e-9)  # Avoid division by zero
        df['vibration_pressure'] = df['vibration'] * df['pressure']

    def _add_delta_features(self, df: pd.DataFrame) -> None:
        """
        Add delta features comparing different time windows.

        Args:
            df: DataFrame with rolling features
        """
        for sensor in self.sensor_columns:
            # Difference between 1h and 6h rolling means
            df[f'{sensor}_delta_mean_1h_6h'] = (
                df[f'{sensor}_rolling_mean_1h'] - df[f'{sensor}_rolling_mean_6h']
            )

    def _add_failure_proximity_features(self, df: pd.DataFrame) -> None:
        """
        Add failure proximity indicators (backward-looking only).

        Args:
            df: DataFrame with failure column
        """
        # Time since last failure (in hours)
        failure_times = df[df['failure'] == 1]['timestamp']
        if not failure_times.empty:
            # For each row, find the most recent failure before current timestamp
            df['hours_since_last_failure'] = df['timestamp'].apply(
                lambda x: (x - failure_times[failure_times <= x].max()).total_seconds() / 3600
                if not failure_times[failure_times <= x].empty else np.nan
            )
        else:
            df['hours_since_last_failure'] = np.nan

        # Fill NaN with large value (no recent failure)
        df['hours_since_last_failure'] = df['hours_since_last_failure'].fillna(9999)

    def _handle_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle NaN values created by rolling and lag operations.

        Args:
            df: DataFrame with potential NaN values

        Returns:
            DataFrame with NaNs handled
        """
        # For lag features, forward fill then fill remaining with 0
        lag_cols = [col for col in df.columns if '_lag_' in col]
        df[lag_cols] = df[lag_cols].ffill().fillna(0)

        # For rolling features, forward fill then fill remaining with 0
        rolling_cols = [col for col in df.columns if 'rolling_' in col or 'ema_' in col]
        df[rolling_cols] = df[rolling_cols].ffill().fillna(0)

        # For rate of change, fill with 0 (no change)
        roc_cols = [col for col in df.columns if '_roc_' in col]
        df[roc_cols] = df[roc_cols].fillna(0)

        # Drop any remaining rows with NaN values (should be minimal)
        initial_shape = df.shape[0]
        df = df.ffill()
        df = df.dropna()
        final_shape = df.shape[0]

        if initial_shape != final_shape:
            logger.warning(f"Dropped {initial_shape - final_shape} rows with NaN values")

        return df

    def fit_scaler(self, df: pd.DataFrame) -> None:
        """
        Fit the standard scaler on training data.

        Args:
            df: Training DataFrame
        """
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'failure', 'failure_24h']]
        self.scaler.fit(df[feature_cols])
        self.is_fitted = True
        logger.info("Fitted StandardScaler on training features")

    def transform_features(self, df: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
        """
        Transform features: create time-series features and optionally scale.

        Args:
            df: Input DataFrame
            scale: Whether to apply standard scaling

        Returns:
            Transformed DataFrame
        """
        df = self.create_time_series_features(df)

        if scale and self.is_fitted:
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'failure', 'failure_24h']]
            df[feature_cols] = self.scaler.transform(df[feature_cols])
            logger.info("Applied StandardScaler transformation")

        return df

    def save_pipeline(self, path: str) -> None:
        """
        Save the feature engineering pipeline for production use.

        Args:
            path: Path to save the pipeline
        """
        ensure_directory(Path(path).parent)
        pipeline_data = {
            'sensor_columns': self.sensor_columns,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        joblib.dump(pipeline_data, path)
        logger.info(f"Saved feature engineering pipeline to {path}")

    def load_pipeline(self, path: str) -> None:
        """
        Load a saved feature engineering pipeline.

        Args:
            path: Path to the saved pipeline
        """
        pipeline_data = joblib.load(path)
        self.sensor_columns = pipeline_data['sensor_columns']
        self.scaler = pipeline_data['scaler']
        self.is_fitted = pipeline_data['is_fitted']
        logger.info(f"Loaded feature engineering pipeline from {path}")

def create_feature_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pipeline_path: str = 'models/feature_pipeline.joblib'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create and apply feature engineering pipeline.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        pipeline_path: Path to save the pipeline

    Returns:
        Tuple of transformed (train_df, val_df)
    """
    engineer = FeatureEngineer()

    # Create features first (without scaling)
    train_features = engineer.create_time_series_features(train_df)
    val_features = engineer.create_time_series_features(val_df)

    # Ensure consistent column order
    feature_cols = sorted([col for col in train_features.columns if col not in ['timestamp', 'failure', 'failure_24h']])
    all_cols = feature_cols + ['timestamp', 'failure', 'failure_24h']

    train_features = train_features[all_cols]
    val_features = val_features[all_cols]

    # Fit scaler on engineered training features
    engineer.fit_scaler(train_features)

    # Apply scaling
    train_features[feature_cols] = engineer.scaler.transform(train_features[feature_cols])
    val_features[feature_cols] = engineer.scaler.transform(val_features[feature_cols])

    # Save pipeline
    engineer.save_pipeline(pipeline_path)

    return train_features, val_features

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import load_and_preprocess_data

    train_df, val_df = load_and_preprocess_data()
    train_features, val_features = create_feature_pipeline(train_df, val_df)

    print(f"Train features shape: {train_features.shape}")
    print(f"Val features shape: {val_features.shape}")
    print("Feature columns:", [col for col in train_features.columns if col not in ['timestamp', 'failure', 'failure_24h']])
