"""
Random Forest Model Training Script with MLflow Experiment Tracking.

Trains a Random Forest classifier for DNS tunnel detection and logs metrics,
parameters, and model artifacts to MLflow.

Usage:
    python scripts/train/train_rf.py --config configs/train_rf.yaml
    python scripts/train/train_rf.py --config configs/train_rf.yaml --run-name "rf-v1"
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.random_forest import DnsRfModel
from utils.logging_setup import safe_log, setup_logger

# Setup logger with UTF-8 support for Windows
logger = setup_logger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary with configuration.

    Raises:
        FileNotFoundError: If config file not found.
        yaml.YAMLError: If YAML parsing fails.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading config from: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        safe_log(logger, "[OK] Config loaded successfully")
        return config
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def validate_class_distribution(
    y: pd.Series,
    min_ratio: float = 0.10,
    context: str = "training",
) -> None:
    """
    Validate that class distribution is balanced enough.
    
    Prevents single-class training set issues that cause:
    - Metrics errors (undefined precision/recall/F1/ROC-AUC)
    - Model performance problems
    
    Args:
        y: Target variable
        min_ratio: Minimum ratio for minority class (default: 10%)
        context: Context name for error message (e.g., 'training', 'validation')
        
    Raises:
        ValueError: If minority class is below threshold
        
    Example:
        validate_class_distribution(y_train, min_ratio=0.20)
        # Raises error if any class < 20% of data
    """
    value_counts = y.value_counts()
    
    if len(value_counts) < 2:
        error_msg = (
            f"[ERROR] {context} set has only 1 class! "
            f"Found: {value_counts.to_dict()}\n"
            f"This causes undefined metrics (Precision/Recall/F1/ROC-AUC).\n\n"
            f"SOLUTION: Run stratified split to balance classes:\n"
            f"  python scripts/data/stratified_split.py --splits-dir data/splits\n\n"
            f"Then update config to use stratified paths:\n"
            f"  configs/train_rf.yaml: train_path='data/splits/train_strat.parquet'"
        )
        raise ValueError(error_msg)
    
    # Check if any class is below minimum ratio
    ratios = value_counts / len(y)
    
    for label, ratio in ratios.items():
        if ratio < min_ratio:
            logger.warning(
                f"[WARNING] {context} set class imbalance: "
                f"class {label} = {ratio:.1%} (below {min_ratio:.0%} threshold)"
            )
    
    logger.info(
        f"[OK] {context} set class distribution: {value_counts.to_dict()}"
    )


def setup_mlflow(config: Dict) -> None:
    """
    Initialize MLflow client and set experiment.

    Args:
        config: Configuration dictionary with mlflow settings.
    """
    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri")
    experiment_name = mlflow_config.get("experiment_name", "DNS-Tunnel-RF")

    # Set tracking URI (local or remote)
    if tracking_uri:
        logger.info(f"Setting MLflow tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
    else:
        logger.info("Using default MLflow tracking (local ./mlruns)")

    # Set experiment
    logger.info(f"Setting experiment: {experiment_name}")
    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow UI: mlflow ui --backend-store-uri file:./mlruns")


def load_data(config: Dict) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load training and validation data from Parquet files.

    Args:
        config: Configuration dictionary with data paths.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val)

    Raises:
        FileNotFoundError: If data files not found.
        KeyError: If required columns missing.
    """
    data_config = config.get("data", {})
    train_path = data_config.get("train_path")
    val_path = data_config.get("val_path")
    target_col = data_config.get("target_col", "label")
    test_size = data_config.get("test_size", 0.2)

    if not train_path:
        raise ValueError("train_path must be specified in config")

    # Load training data
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    logger.info(f"Loading training data from: {train_path}")
    df_train = pd.read_parquet(train_path)
    logger.info(f"  Loaded {len(df_train):,} records, {len(df_train.columns)} columns")

    # Extract features and target
    if target_col not in df_train.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")

    y_train = df_train[target_col].astype("int8")
    
    # Select feature columns
    exclude_cols = {target_col, "timestamp", "session_id", "split"}  # Non-numeric metadata cols
    if data_config.get("feature_cols"):
        # Use specified feature columns
        X_train = df_train[data_config["feature_cols"]].copy()
    else:
        # Auto-detect: all columns except target and non-numeric metadata
        X_train = df_train.drop(columns=[c for c in df_train.columns if c in exclude_cols])
        # Further filter: only select numeric columns
        X_train = X_train.select_dtypes(include=["float64", "float32", "int64", "int32", "int16", "int8"])

    logger.info(f"Training set: X shape {X_train.shape}, y shape {y_train.shape}")
    logger.info(f"  Features: {list(X_train.columns)}")
    
    # Validate class distribution (will raise error if only 1 class)
    validate_class_distribution(y_train, min_ratio=0.10, context="training")
    
    logger.info(f"  Class distribution: {y_train.value_counts().to_dict()}")

    # Load validation data if provided, otherwise split training data
    if val_path:
        val_path = Path(val_path)
        if not val_path.exists():
            raise FileNotFoundError(f"Validation data not found: {val_path}")

        logger.info(f"Loading validation data from: {val_path}")
        df_val = pd.read_parquet(val_path)

        if target_col not in df_val.columns:
            raise KeyError(f"Target column '{target_col}' not found in validation data")

        y_val = df_val[target_col].astype("int8")
        
        # Select same feature columns as training
        if data_config.get("feature_cols"):
            X_val = df_val[data_config["feature_cols"]].copy()
        else:
            X_val = df_val.drop(columns=[c for c in df_val.columns if c in exclude_cols])
            X_val = X_val.select_dtypes(include=["float64", "float32", "int64", "int32", "int16", "int8"])
        
        # Ensure same columns as training
        X_val = X_val[X_train.columns]
    else:
        # Split training data for validation
        logger.info(f"Splitting training data (test_size={test_size})...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=test_size, random_state=42, stratify=y_train
        )

    logger.info(f"Validation set: X shape {X_val.shape}, y shape {y_val.shape}")
    
    # Validate class distribution
    validate_class_distribution(y_val, min_ratio=0.05, context="validation")
    
    logger.info(f"  Class distribution: {y_val.value_counts().to_dict()}")

    return X_train, y_train, X_val, y_val


def save_confusion_matrix_plot(
    model: DnsRfModel,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    filename: str = "confusion_matrix.png",
) -> Path:
    """
    Generate and save confusion matrix visualization.

    Args:
        model: Trained DnsRfModel instance.
        X: Feature matrix.
        y: True labels.
        output_dir: Directory to save plot.
        filename: Output filename.

    Returns:
        Path to saved plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    cm = model.get_confusion_matrix(X, y)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    safe_log(logger, f"[OK] Saved confusion matrix plot: {output_path}")
    return output_path


def save_feature_importance_plot(
    model: DnsRfModel,
    X: pd.DataFrame,
    output_dir: Path,
    filename: str = "feature_importance.png",
) -> Path:
    """
    Generate and save feature importance visualization.

    Args:
        model: Trained DnsRfModel instance.
        X: Feature matrix (used for column names).
        output_dir: Directory to save plot.
        filename: Output filename.

    Returns:
        Path to saved plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    importances, _ = model.get_feature_importance()

    # Get feature names
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    # Sort by importance
    indices = np.argsort(importances)[::-1][:10]  # Top 10

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="#2ecc71",
        edgecolor="black",
    )
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Top 10 Feature Importance", fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    safe_log(logger, f"[OK] Saved feature importance plot: {output_path}")
    return output_path


def train_model(
    config: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> DnsRfModel:
    """
    Train Random Forest model with MLflow tracking.

    Args:
        config: Configuration dictionary.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Trained DnsRfModel instance.
    """
    model_config = config.get("model", {})
    output_config = config.get("output", {})
    mlflow_config = config.get("mlflow", {})

    models_dir = Path(output_config.get("models_dir", "models/"))
    plots_dir = Path(output_config.get("plots_dir", "plots/"))
    save_plots = output_config.get("save_plots", True)

    # Start MLflow run
    run_name = mlflow_config.get("run_name")
    with mlflow.start_run(run_name=run_name):
        logger.info("=" * 60)
        logger.info("Started MLflow Run")
        logger.info("=" * 60)

        # Log parameters
        logger.info("Logging parameters to MLflow...")
        mlflow.log_params(model_config)

        # Initialize and train model
        logger.info("Initializing Random Forest model...")
        model = DnsRfModel(**model_config)

        logger.info("Training model...")
        model.fit(X_train, y_train)

        # Evaluate on training set
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SET METRICS")
        logger.info("=" * 60)
        train_metrics = model.evaluate(X_train, y_train, set_name="training")

        # Prefix train metrics for MLflow (skip NaN values)
        train_metrics_prefixed = {f"train_{k}": v for k, v in train_metrics.items() if not np.isnan(v)}
        mlflow.log_metrics(train_metrics_prefixed)

        # Evaluate on validation set
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SET METRICS")
        logger.info("=" * 60)
        val_metrics = model.evaluate(X_val, y_val, set_name="validation")

        # Prefix validation metrics for MLflow (skip NaN values)
        val_metrics_prefixed = {f"val_{k}": v for k, v in val_metrics.items() if not np.isnan(v)}
        mlflow.log_metrics(val_metrics_prefixed)

        # Save model
        logger.info(f"\nSaving model to {models_dir}...")
        model_path = models_dir / "random_forest.pkl"
        model.save(str(model_path))

        # Log model to MLflow
        logger.info("Logging model to MLflow...")
        mlflow.sklearn.log_model(
            model.model,
            "random_forest",
            registered_model_name="dns-tunnel-rf",
        )

        # Save and log plots
        if save_plots:
            logger.info("\nGenerating plots...")

            # Confusion matrix
            cm_path = save_confusion_matrix_plot(model, X_val, y_val, plots_dir, "cm_validation.png")
            mlflow.log_artifact(str(cm_path), artifact_path="plots")

            # Feature importance
            fi_path = save_feature_importance_plot(model, X_train, plots_dir, "feature_importance.png")
            mlflow.log_artifact(str(fi_path), artifact_path="plots")

        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Training accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Validation F1-Score: {val_metrics['f1']:.4f}")
        logger.info(f"Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
        logger.info(f"Model saved to: {model_path}")
        logger.info("=" * 60)

        return model


def main() -> int:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Random Forest model with MLflow tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML)",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional MLflow run name (default: auto-generated)",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Note: Logger already configured with UTF-8 support in module import
        # File logging is handled automatically by setup_logger

        # Setup MLflow
        setup_mlflow(config)

        # Override run name if provided
        if args.run_name:
            config["mlflow"]["run_name"] = args.run_name

        # Load data
        X_train, y_train, X_val, y_val = load_data(config)

        # Train model
        model = train_model(config, X_train, y_train, X_val, y_val)

        safe_log(logger, "[OK] Training completed successfully")
        safe_log(logger, "View results: mlflow ui --backend-store-uri file:./mlruns")

        return 0

    except ValueError as e:
        # Class distribution or validation errors
        logger.error(f"Validation failed: {e}")
        return 1
    except FileNotFoundError as e:
        # Missing files (config, data)
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        # All other errors
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
