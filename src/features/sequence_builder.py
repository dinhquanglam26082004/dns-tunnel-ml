"""
LSTM Sequence Building Module.

Constructs sequences from DNS queries for recurrent neural networks.
Groups queries by session (src_ip + base_domain), applies sliding window,
and handles temporal normalization with scaler isolation to prevent data leakage.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


def _pad_sequence(seq: np.ndarray, target_len: int, pad_value: float = 0.0) -> np.ndarray:
    """
    Pad or truncate sequence to target length.

    Args:
        seq: Input sequence of shape (seq_len, n_features)
        target_len: Desired sequence length
        pad_value: Value to use for padding

    Returns:
        Sequence of shape (target_len, n_features)
    """
    current_len = seq.shape[0]

    if current_len >= target_len:
        # Truncate: take first target_len items
        return seq[:target_len, :]
    else:
        # Pad: add pad_value rows at start
        n_features = seq.shape[1]
        padding = np.full((target_len - current_len, n_features), pad_value, dtype=seq.dtype)
        return np.vstack([padding, seq])


def _sliding_window(
    session_data: np.ndarray,
    session_labels: np.ndarray,
    seq_len: int,
    stride: int,
) -> Tuple[list, list]:
    """
    Extract sliding windows from session data.

    Args:
        session_data: Features of shape (T, n_features)
        session_labels: Binary labels of shape (T,)
        seq_len: Window length
        stride: Step size between windows

    Returns:
        Tuple of (windows_list, window_labels_list)
        Each window: (seq_len, n_features)
    """
    windows = []
    labels = []

    T = session_data.shape[0]

    # Handle case where session is shorter than seq_len
    if T < seq_len:
        padded = _pad_sequence(session_data, seq_len)
        # Label: mean of session labels >= 0.5 → 1, else 0
        label = 1 if session_labels.mean() >= 0.5 else 0
        windows.append(padded)
        labels.append(label)
    else:
        # Sliding window over longer sequences
        for start_idx in range(0, T - seq_len + 1, stride):
            end_idx = start_idx + seq_len
            window = session_data[start_idx:end_idx, :]
            window_labels = session_labels[start_idx:end_idx]

            # Label: majority voting (>= 0.5 → 1)
            label = 1 if window_labels.mean() >= 0.5 else 0

            windows.append(window)
            labels.append(label)

    return windows, labels


def build_lstm_sequences(
    df: pd.DataFrame,
    seq_len: int = 30,
    stride: int = 15,
    scaler: Optional[RobustScaler] = None,
    fit_scaler: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[RobustScaler]]:
    """
    Build LSTM-ready sequences from DNS features.

    Groups queries by session (src_ip + base_domain), sorts by timestamp,
    applies sliding window with padding/truncation, and normalizes IAT features.

    Args:
        df: Features DataFrame with columns:
            - qname_entropy (float)
            - qname_length (int)
            - numeric_ratio (float)
            - subdomain_depth (int)
            - qtype (int)
            - iat_seconds (float) - will be normalized
            - label (int) - 0/1
            - src_ip (str) - for session grouping
            - base_domain (str) - for session grouping
            - timestamp (float) - for sorting within session

        seq_len: Length of sequences (default 30 queries)
        stride: Step size for sliding window (default 15)
        scaler: Pre-fitted RobustScaler for IAT normalization.
            If None and fit_scaler=True, fits new scaler on IAT data.
            If None and fit_scaler=False, uses identity (no scaling).
        fit_scaler: Whether to fit a new scaler if none provided

    Returns:
        Tuple of (X, y, scaler):
        - X: Features array of shape (N_windows, seq_len, n_features)
        - y: Labels array of shape (N_windows,) with 0/1 values
        - scaler: Fitted RobustScaler (or None if not used)

    Raises:
        ValueError: If required columns missing or no valid sequences
    """
    logger.info(f"Building LSTM sequences from {len(df)} records")

    # Required columns
    required_features = [
        "qname_entropy",
        "qname_length",
        "numeric_ratio",
        "subdomain_depth",
        "qtype",
        "iat_seconds",
        "label",
        "src_ip",
        "base_domain",
    ]

    missing = set(required_features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure timestamp for sorting (optional, use index if not present)
    if "timestamp" not in df.columns:
        logger.warning("No 'timestamp'; using row order for session sorting")
        df = df.copy()
        df["timestamp"] = np.arange(len(df))

    # Feature columns (exclude src_ip, base_domain used for grouping)
    feature_cols = [
        "qname_entropy",
        "qname_length",
        "numeric_ratio",
        "subdomain_depth",
        "qtype",
        "iat_seconds",
    ]

    n_features = len(feature_cols)

    # Fit scaler if requested (on training data only to avoid leakage)
    if fit_scaler and scaler is None:
        logger.info("Fitting RobustScaler on IAT features")
        iat_data = df["iat_seconds"].values.reshape(-1, 1)
        scaler = RobustScaler()
        scaler.fit(iat_data)

    # Collect all windows
    all_windows = []
    all_labels = []

    # Group by session (src_ip + base_domain)
    grouped = df.groupby(["src_ip", "base_domain"], sort=False)
    n_sessions = len(grouped)
    logger.info(f"Processing {n_sessions} sessions")

    for session_idx, (group_key, session_df) in enumerate(grouped):
        if (session_idx + 1) % 1000 == 0:
            logger.debug(f"Processed {session_idx + 1}/{n_sessions} sessions")

        # Sort by timestamp within session
        session_df = session_df.sort_values("timestamp")

        # Extract feature data and labels
        X_session = session_df[feature_cols].values.astype(np.float32)
        y_session = session_df["label"].values.astype(np.int8)

        # Normalize IAT (column index 5 = iat_seconds)
        if scaler is not None:
            iat_col_idx = feature_cols.index("iat_seconds")
            iat_values = X_session[:, iat_col_idx].reshape(-1, 1)
            X_session[:, iat_col_idx] = scaler.transform(iat_values).ravel()

        # Extract sliding windows
        windows, labels = _sliding_window(X_session, y_session, seq_len, stride)

        all_windows.extend(windows)
        all_labels.extend(labels)

    if not all_windows:
        raise ValueError("No valid sequences generated")

    # Stack into arrays
    X = np.array(all_windows, dtype=np.float32)  # (N, seq_len, n_features)
    y = np.array(all_labels, dtype=np.int8)  # (N,)

    logger.info(f"Generated {len(X)} sequences: X shape {X.shape}, y shape {y.shape}")

    # Log class distribution
    n_class_1 = np.sum(y)
    logger.info(f"Class distribution: {len(y) - n_class_1} benign, {n_class_1} malicious")

    return X, y, scaler


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    sample_data = pd.DataFrame(
        {
            "qname_entropy": np.random.uniform(2, 7, 100),
            "qname_length": np.random.randint(10, 200, 100),
            "numeric_ratio": np.random.uniform(0, 0.3, 100),
            "subdomain_depth": np.random.randint(1, 5, 100),
            "qtype": np.random.choice([1, 28], 100),
            "iat_seconds": np.random.exponential(1.0, 100),
            "label": np.random.randint(0, 2, 100),
            "src_ip": ["192.168.1." + str(i % 10) for i in range(100)],
            "base_domain": ["example.com" if i % 2 == 0 else "other.com" for i in range(100)],
            "timestamp": np.arange(100) + np.random.uniform(0, 0.1, 100),
        }
    )

    X, y, scaler = build_lstm_sequences(sample_data, seq_len=10, stride=5, fit_scaler=True)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Scaler mean: {scaler.center_}, Scaler scale: {scaler.scale_}")
