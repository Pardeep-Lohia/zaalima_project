import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
from pathlib import Path
import logging

from .utils import setup_logging, ensure_directory, get_project_root

logger = setup_logging()

def generate_synthetic_iot_data(
    n_samples: int = 10000,
    failure_rate: float = 0.005,
    start_date: str = '2023-01-01',
    freq: str = '1H',
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic IoT sensor data for predictive maintenance.

    Args:
        n_samples: Number of data points to generate
        failure_rate: Probability of failure (< 1%)
        start_date: Start date for time series
        freq: Frequency of data points (e.g., '1H' for hourly)
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with timestamp, sensor readings, and failure labels
    """
    np.random.seed(random_seed)

    # Generate timestamps
    freq : str = '1h'
    timestamps = pd.date_range(start=start_date, periods=n_samples, freq=freq)

    # Generate sensor data with normal operation patterns
    # Temperature: Normal range 20-80°C with some variation
    temperature_base = 50 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24)  # Daily cycle
    temperature_noise = np.random.normal(0, 2, n_samples)
    temperature = temperature_base + temperature_noise

    # Vibration: Normal range 0-10 units
    vibration_base = 2 + 0.5 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    vibration_noise = np.random.exponential(0.5, n_samples)
    vibration = vibration_base + vibration_noise

    # Pressure: Normal range 90-110 units
    pressure_base = 100 + 2 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    pressure_noise = np.random.normal(0, 1, n_samples)
    pressure = pressure_base + pressure_noise

    # Generate failure labels (extremely rare)
    failure = np.random.binomial(1, failure_rate, n_samples)

    # For failure cases, modify sensor readings to show realistic precursors
    failure_indices = np.where(failure == 1)[0]

    # Introduce gradual degradation patterns 12-48 hours before failure
    for idx in failure_indices:
        # Random precursor window between 12-48 hours
        precursor_hours = np.random.randint(12, 48)
        anomaly_start = max(0, idx - precursor_hours)

        # Temperature drift: Gradual increase over precursor period (wear and tear)
        temp_drift = np.random.uniform(0.1, 0.3)  # °C per hour drift
        temp_anomaly = np.cumsum(np.full(idx - anomaly_start + 1, temp_drift))
        temperature[anomaly_start:idx+1] += temp_anomaly

        # Sustained high vibration: Base increase with periodic spikes
        vib_base_increase = np.random.uniform(2, 4)  # 2-4 units sustained increase
        vibration[anomaly_start:idx+1] += vib_base_increase
        # Add periodic vibration spikes (mechanical stress indicators)
        spike_frequency = np.random.uniform(0.2, 0.4)  # 20-40% chance per hour
        for i in range(anomaly_start, idx+1):
            if np.random.random() < spike_frequency:
                vibration[i] += np.random.uniform(1, 3)  # Additional spike

        # Pressure instability: Increasing variance over time
        # Start with small variations, increase towards failure
        pressure_std_start = 0.5
        pressure_std_end = np.random.uniform(3, 6)
        pressure_stds = np.linspace(pressure_std_start, pressure_std_end, idx - anomaly_start + 1)
        for i, std_val in enumerate(pressure_stds):
            pressure[anomaly_start + i] += np.random.normal(0, std_val)

        # Rising rolling variance: Add additional noise that increases variance
        variance_increase = np.linspace(0, np.random.uniform(2, 4), idx - anomaly_start + 1)
        for i, var_inc in enumerate(variance_increase):
            pressure[anomaly_start + i] += np.random.normal(0, var_inc * 0.5)

        # Sudden pressure drop at failure (final system failure)
        pressure[idx] -= np.random.uniform(15, 25)

        # Ensure gradual appearance - add smoothing to prevent instantaneous changes
        # Smooth the transitions at anomaly_start to avoid sharp edges
        if anomaly_start > 0:
            transition_window = min(6, idx - anomaly_start)  # 6-hour transition
            for sensor, name in [(temperature, 'temp'), (vibration, 'vib'), (pressure, 'press')]:
                # Linear interpolation for smooth transition
                start_val = sensor[anomaly_start - 1] if anomaly_start > 0 else sensor[anomaly_start]
                end_val = sensor[anomaly_start]
                transition_vals = np.linspace(start_val, end_val, transition_window + 1)[1:]
                sensor[anomaly_start:anomaly_start + transition_window] = transition_vals

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'vibration': vibration,
        'pressure': pressure,
        'failure': failure
    })

    logger.info(f"Generated synthetic dataset with {n_samples} samples, {failure.sum()} failures ({failure_rate*100:.2f}%)")

    return df

def time_aware_train_val_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.3,
    time_column: str = 'timestamp'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform time-aware train/validation split to prevent data leakage.

    Args:
        df: Input DataFrame sorted by time
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        time_column: Name of timestamp column

    Returns:
        Tuple of (train_df, val_df)
    """
    if not df[time_column].is_monotonic_increasing:
        df = df.sort_values(time_column).reset_index(drop=True)
        logger.warning("Data was not sorted by time, sorting now")

    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Ensure we don't exceed total
    if n_train + n_val > n_total:
        n_val = n_total - n_train

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()

    logger.info(f"Time-aware split: Train {len(train_df)} samples, Val {len(val_df)} samples")
    logger.info(f"Train period: {train_df[time_column].min()} to {train_df[time_column].max()}")
    logger.info(f"Val period: {val_df[time_column].min()} to {val_df[time_column].max()}")

    return train_df, val_df

def load_and_preprocess_data(
    data_path: Optional[str] = None,
    save_processed: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data and perform preprocessing with time-aware split.

    Args:
        data_path: Path to existing data file (if None, generate synthetic)
        save_processed: Whether to save processed data

    Returns:
        Tuple of (train_df, val_df)
    """
    if data_path is None:
        # Generate synthetic data
        logger.info("No data path provided, generating synthetic IoT data")
        df = generate_synthetic_iot_data()

        if save_processed:
            data_dir = Path(get_project_root()) / 'data'
            ensure_directory(data_dir)
            df.to_csv(data_dir / 'synthetic_iot_data.csv', index=False)
            logger.info("Saved synthetic data to data/synthetic_iot_data.csv")
    else:
        # Load existing data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create shifted target for 24-hour ahead prediction
    df['failure_24h'] = df['failure'].shift(-24)

    # Drop rows with NaN in failure_24h (last 24 rows)
    initial_shape = df.shape[0]
    df = df.dropna(subset=['failure_24h'])
    final_shape = df.shape[0]
    dropped_rows = initial_shape - final_shape

    # Log remaining positive samples
    positive_samples = df['failure_24h'].sum()
    logger.info(f"After shifting target 24 hours ahead: dropped {dropped_rows} rows, {positive_samples} positive samples remain")

    # Perform time-aware split AFTER shifting target
    train_df, val_df = time_aware_train_val_split(df)

    if save_processed:
        data_dir = Path(get_project_root()) / 'data'
        ensure_directory(data_dir)
        train_df.to_csv(data_dir / 'train_raw.csv', index=False)
        val_df.to_csv(data_dir / 'val_raw.csv', index=False)
        logger.info("Saved processed train/val splits")

    return train_df, val_df

def analyze_class_distribution(df: pd.DataFrame, title: str = "Dataset") -> None:
    """
    Analyze and log class distribution in the dataset.

    Args:
        df: DataFrame to analyze
        title: Title for the analysis
    """
    target_col = 'failure_24h' if 'failure_24h' in df.columns else 'failure'

    failure_count = df[target_col].sum()
    total_count = len(df)
    failure_rate = failure_count / total_count

    logger.info(f"{title} Analysis ({target_col}):")
    logger.info(f"  Total samples: {total_count}")
    logger.info(f"  Positive samples: {failure_count}")
    logger.info(f"  Positive rate: {failure_rate:.4f} ({failure_rate*100:.2f}%)")
    logger.info(f"  Negative samples: {total_count - failure_count}")

if __name__ == "__main__":
    # Generate and preprocess data
    train_df, val_df = load_and_preprocess_data()

    # Analyze distributions
    analyze_class_distribution(train_df, "Training Set")
    analyze_class_distribution(val_df, "Validation Set")

    print("Data preprocessing completed!")
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
