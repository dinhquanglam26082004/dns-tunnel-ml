"""
CLI script for feature extraction and LSTM sequence building.

Usage:
    python scripts/features/extract.py \
        --input data/processed/clean.parquet \
        --output-features data/splits/features.parquet \
        --output-sequences data/splits/sequences/ \
        --seq-len 30 \
        --stride 15
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from features.extractor import extract_dns_features
from features.sequence_builder import build_lstm_sequences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_input(input_path: Path) -> bool:
    """Validate input file exists and is readable"""
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return False

    if input_path.suffix not in [".parquet", ".csv", ".pq"]:
        logger.error(f"Unsupported file format: {input_path.suffix}")
        return False

    return True


def load_data(input_path: Path) -> pd.DataFrame:
    """Load data from Parquet or CSV"""
    logger.info(f"Loading data from: {input_path}")

    try:
        if input_path.suffix == ".parquet" or input_path.suffix == ".pq":
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)

        logger.info(f"Loaded {len(df)} records, {len(df.columns)} columns")
        return df

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract DNS features from raw data"""
    logger.info("Extracting DNS features...")

    try:
        features_df = extract_dns_features(df)
        logger.info(f"Extracted features for {len(features_df)} records")
        return features_df

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise


def build_sequences(
    features_df: pd.DataFrame,
    seq_len: int,
    stride: int,
) -> tuple:
    """Build LSTM sequences"""
    logger.info(f"Building LSTM sequences (seq_len={seq_len}, stride={stride})...")

    try:
        X, y, scaler = build_lstm_sequences(
            features_df,
            seq_len=seq_len,
            stride=stride,
            fit_scaler=True,
        )

        logger.info(f"Built {len(X)} sequences")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y shape: {y.shape}")
        logger.info(f"  Class distribution: {np.bincount(y)}")

        return X, y, scaler

    except Exception as e:
        logger.error(f"Sequence building failed: {e}")
        raise


def save_features(features_df: pd.DataFrame, output_path: Path) -> None:
    """Save extracted features to Parquet"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving features to: {output_path}")

    try:
        features_df.to_parquet(output_path, index=False)
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ Saved features ({file_size_mb:.2f} MB)")

    except Exception as e:
        logger.error(f"Failed to save features: {e}")
        raise


def save_sequences(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
) -> None:
    """Save sequences and labels to numpy files"""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving sequences to: {output_dir}")

    try:
        # Save X and y as numpy files
        X_path = output_dir / "X_sequences.npy"
        y_path = output_dir / "y_sequences.npy"

        np.save(X_path, X)
        np.save(y_path, y)

        X_size_mb = X_path.stat().st_size / 1024 / 1024
        y_size_mb = y_path.stat().st_size / 1024 / 1024

        logger.info(f"✓ Saved X_sequences.npy ({X_size_mb:.2f} MB)")
        logger.info(f"✓ Saved y_sequences.npy ({y_size_mb:.2f} MB)")

    except Exception as e:
        logger.error(f"Failed to save sequences: {e}")
        raise


def save_scaler(
    scaler,
    output_dir: Path,
) -> None:
    """Save fitted RobustScaler to pickle file"""
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = output_dir / "scaler.pkl"

    logger.info(f"Saving scaler to: {scaler_path}")

    try:
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        logger.info(f"✓ Saved scaler.pkl")

    except Exception as e:
        logger.error(f"Failed to save scaler: {e}")
        raise


def main() -> int:
    """Main extraction pipeline"""
    parser = argparse.ArgumentParser(
        description="Extract DNS features and build LSTM sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input Parquet/CSV file path",
    )

    parser.add_argument(
        "--output-features",
        type=Path,
        help="Output features Parquet file path (optional)",
    )

    parser.add_argument(
        "--output-sequences",
        type=Path,
        help="Output sequences directory (optional)",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=30,
        help="LSTM sequence length (default: 30)",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Sliding window stride (default: 15)",
    )

    parser.add_argument(
        "--save-scaler",
        action="store_true",
        help="Save fitted RobustScaler to pickle file",
    )

    args = parser.parse_args()

    # Validate inputs
    if not validate_input(args.input):
        return 1

    # Check that at least one output is specified
    if not args.output_features and not args.output_sequences:
        logger.error("Specify at least one output: --output-features or --output-sequences")
        return 1

    # Load data
    try:
        df = load_data(args.input)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return 1

    # Extract features
    try:
        features_df = extract_features(df)
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return 1

    # Save features if requested
    if args.output_features:
        try:
            save_features(features_df, args.output_features)
        except Exception as e:
            logger.error(f"Failed to save features: {e}")
            return 1

    # Build sequences if requested
    scaler = None
    if args.output_sequences:
        try:
            X, y, scaler = build_sequences(
                features_df,
                seq_len=args.seq_len,
                stride=args.stride,
            )

            save_sequences(X, y, args.output_sequences)

            # Save scaler if requested
            if args.save_scaler and scaler is not None:
                save_scaler(scaler, args.output_sequences)

        except Exception as e:
            logger.error(f"Sequence building failed: {e}")
            return 1

    logger.info("=" * 60)
    logger.info("Feature extraction and sequence building complete!")
    logger.info("=" * 60)

    if args.output_features:
        logger.info(f"Features saved to: {args.output_features}")

    if args.output_sequences:
        logger.info(f"Sequences saved to: {args.output_sequences}/")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
