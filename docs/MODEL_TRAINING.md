# Random Forest Model Training & MLflow Setup

Complete guide for training DNS Tunnel Random Forest model with MLflow experiment tracking.

## 🎯 Overview

This pipeline provides:
- **Production-ready model wrapper** with scikit-learn's RandomForestClassifier
- **MLflow experiment tracking** for reproducible model development
- **Configurable hyperparameters** via YAML configuration
- **Comprehensive metrics logging** (Accuracy, F1, ROC-AUC, Precision, Recall)
- **Automatic visualization** (confusion matrix, feature importance)

## 📦 Files

### 1. Model Wrapper: `src/models/random_forest.py`
- **Class:** `DnsRfModel`
- **Methods:**
  - `fit(X, y)` - Train on data
  - `predict(X)` - Make predictions
  - `predict_proba(X)` - Get class probabilities
  - `evaluate(X, y, set_name)` - Compute metrics
  - `get_confusion_matrix(X, y)` - Generate confusion matrix
  - `save(path)` - Serialize model with joblib
  - `load(path)` - Deserialize model

### 2. Configuration: `configs/train_rf.yaml`
- **model** section: Hyperparameters (n_estimators, max_depth, class_weight, etc.)
- **data** section: Input paths, target column, test split
- **mlflow** section: Tracking URI, experiment name, tags
- **output** section: Models directory, plots, save options
- **logging** section: Log level and file

### 3. Training Script: `scripts/train/train_rf.py`
- **CLI tool** with argparse
- **Config loading** via YAML
- **Data loading** from Parquet files
- **MLflow integration** for tracking
- **Automatic visualization** generation
- **Comprehensive logging**

## 🚀 Quick Start

### 1. Install MLflow (Already in pyproject.toml)

```bash
# If not installed
pip install mlflow>=2.0.0
```

### 2. Create Sample Data

If you don't have data yet, generate sample features:

```bash
# This creates data/processed/sample.parquet
python << 'EOF'
import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

df = pd.DataFrame({
    'qname_entropy': np.random.uniform(2, 7, n),
    'qname_length': np.random.randint(10, 200, n),
    'numeric_ratio': np.random.uniform(0, 0.5, n),
    'subdomain_depth': np.random.randint(1, 5, n),
    'qtype': np.random.choice([1, 28], n),
    'iat_seconds': np.random.exponential(1.0, n),
    'label': np.random.choice([0, 1], n, p=[0.8, 0.2])
})

df.to_parquet('data/processed/sample.parquet')
print(f"Created sample data: {df.shape}")
EOF
```

Update config: Change `data.train_path` to `"data/processed/sample.parquet"`

### 3. Run Training

```bash
# Activate environment
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate.bat         # Windows

# Train with default config
python scripts/train/train_rf.py --config configs/train_rf.yaml

# Or with custom run name
python scripts/train/train_rf.py --config configs/train_rf.yaml --run-name "rf-baseline-v1"
```

**Output:**
```
INFO:__main__:Loading config from: configs/train_rf.yaml
INFO:__main__:✓ Config loaded successfully
INFO:__main__:Setting experiment: DNS-Tunnel-RF
INFO:__main__:Loading training data from: data/splits/features.parquet
INFO:__main__:Training set: X shape (1600, 6), y shape (1600,)
...
INFO:__main__:============================================================
INFO:__main__:VALIDATION SET METRICS
INFO:__main__:============================================================
INFO:__main__:val_accuracy: 0.8750
INFO:__main__:val_precision: 0.8286
INFO:__main__:val_f1: 0.7895
INFO:__main__:val_roc_auc: 0.9154
...
```

### 4. View MLflow Dashboard

```bash
# Start UI server (runs on http://localhost:5000)
mlflow ui

# Or specify tracking URI explicitly
mlflow ui --backend-store-uri file:./mlruns
```

Then open browser: http://localhost:5000

## 📊 MLflow Dashboard Features

### Metrics Tracked
- **train_accuracy**: Training set accuracy
- **train_precision**: Training set precision (malicious class)
- **train_recall**: Training set recall (malicious class)
- **train_f1**: Training set F1-score
- **train_roc_auc**: Training set ROC-AUC
- **val_accuracy**: Validation set accuracy
- **val_precision**: Validation set precision
- **val_recall**: Validation set recall
- **val_f1**: Validation set F1-score
- **val_roc_auc**: Validation set ROC-AUC

### Parameters Logged
- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- class_weight
- random_state
- n_jobs

### Artifacts Saved
- `random_forest.pkl` - Model pickle file
- `cm_validation.png` - Confusion matrix visualization
- `feature_importance.png` - Top 10 feature importances

## 🔧 Configuration

### Typical Configurations

**Baseline (Balanced, Deeper Trees):**
```yaml
model:
  n_estimators: 100
  max_depth: 20
  min_samples_split: 5
  class_weight: "balanced"
```

**Fast Training (Shallow Trees):**
```yaml
model:
  n_estimators: 50
  max_depth: 10
  min_samples_split: 10
  class_weight: null
```

**Complex Model (Potential Overfitting):**
```yaml
model:
  n_estimators: 500
  max_depth: 30
  min_samples_split: 2
  class_weight: "balanced"
```

**Hyperparameter Tuning Template:**
```bash
#!/bin/bash
# Run multiple configurations

for n_est in 50 100 200; do
  for depth in 10 20 30; do
    python scripts/train/train_rf.py \
      --config configs/train_rf.yaml \
      --run-name "rf-tune-ne${n_est}-d${depth}"
  done
done

# Compare in MLflow UI and pick best
mlflow ui
```

## 📈 MLflow Tracking Backends

### Local File System (Default)

```yaml
# configs/train_rf.yaml
mlflow:
  tracking_uri: null    # Uses ./mlruns directory
  experiment_name: "DNS-Tunnel-RF"
```

**Enable:**
```bash
# No setup needed, automatically creates ./mlruns/
python scripts/train/train_rf.py --config configs/train_rf.yaml
```

### Remote MLflow Server

**1. Start MLflow server:**
```bash
# On server machine
mlflow server --backend-store-uri ./mlruns --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000
```

**2. Update config:**
```yaml
# configs/train_rf.yaml
mlflow:
  tracking_uri: "http://192.168.1.100:5000"
  experiment_name: "DNS-Tunnel-RF"
```

**3. Train:**
```bash
# On client machine
python scripts/train/train_rf.py --config configs/train_rf.yaml
```

**4. View:**
```bash
# Access on browser
http://192.168.1.100:5000
```

### Using Environment Variables

```bash
# Set tracking URI via environment
export MLFLOW_TRACKING_URI="file:./mlruns"
export MLFLOW_EXPERIMENT_NAME="DNS-Tunnel-RF"

# Then config can use null for both:
# tracking_uri: null
# experiment_name: null

python scripts/train/train_rf.py --config configs/train_rf.yaml
```

## 🔍 Inspect Model Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# List registered models
for model_info in client.search_registered_models():
    print(f"Model: {model_info.name}")
    print(f"  Stage: {model_info.latest_versions[0].current_stage}")

# Get best run in experiment
experiment = client.get_experiment_by_name("DNS-Tunnel-RF")
runs = client.search_runs(experiment_ids=[experiment.experiment_id])

best_run = max(runs, key=lambda r: r.data.metrics.get("val_f1", 0))
print(f"Best run: {best_run.info.run_id}")
print(f"Metrics: {best_run.data.metrics}")
```

## 📊 Python API

### Load and Use Trained Model

```python
from src.models.random_forest import DnsRfModel
import pandas as pd

# Load model
model = DnsRfModel.load("models/random_forest.pkl")

# Make predictions
X_test = pd.read_parquet("data/test/features.parquet")
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate
y_test = pd.read_parquet("data/test/labels.parquet")
metrics = model.evaluate(X_test, y_test)
print(f"Validation F1: {metrics['f1']:.4f}")
```

### Access MLflow Run Data

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get latest run
experiment = client.get_experiment_by_name("DNS-Tunnel-RF")
run = client.search_runs(experiment_ids=[experiment.experiment_id])[0]

run_id = run.info.run_id
print(f"Run ID: {run_id}")
print(f"Metrics: {run.data.metrics}")
print(f"Parameters: {run.data.params}")

# Download artifacts
mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="plots",
    dst_path="./downloaded_plots"
)
```

## 🐛 Troubleshooting

### Issue: "No such file or directory: data/splits/features.parquet"

**Solution:**
```bash
# Generate data first
python scripts/features/extract.py \
    --input data/processed/raw.parquet \
    --output-features data/splits/features.parquet

# Or create sample data
python scripts/data/generate_sample.py
```

### Issue: "ModuleNotFoundError: No module named 'mlflow'"

**Solution:**
```bash
# Install MLflow
pip install mlflow>=2.0.0

# Or reinstall with dev dependencies
pip install -e ".[dev,tracking]"
```

### Issue: "YAML parsing error"

**Solution:**
- Check YAML indentation (must be consistent)
- Verify no tabs in config file
- Use `yamllint` to validate:
  ```bash
  pip install yamllint
  yamllint configs/train_rf.yaml
  ```

### Issue: "ValueError: target column 'label' not found"

**Solution:**
- Check data file has 'label' column
```bash
python << 'EOF'
import pandas as pd
df = pd.read_parquet("data/splits/features.parquet")
print(df.columns)
print(df['label'].value_counts())
EOF
```

### Issue: MLflow UI shows no runs

**Solution:**
```bash
# Check tracking URI
ls -la mlruns/

# Or restart UI
pkill -f mlflow
mlflow ui --backend-store-uri file:./mlruns
```

## 📚 Integration with Other Components

### Feature Engineering Pipeline

```bash
# 1. Extract features
python scripts/features/extract.py \
    --input data/processed/raw.parquet \
    --output-features data/splits/features.parquet

# 2. Train model
python scripts/train/train_rf.py --config configs/train_rf.yaml

# 3. Evaluate
python scripts/eval/evaluate.py --predictions data/predictions.csv
```

### DVC Pipeline

Add to `dvc.yaml`:

```yaml
stages:
  train_rf:
    cmd: python scripts/train/train_rf.py --config configs/train_rf.yaml
    deps:
      - scripts/train/train_rf.py
      - src/models/random_forest.py
      - configs/train_rf.yaml
      - data/splits/features.parquet
    outs:
      - models/random_forest.pkl
    metrics:
      - plots/training_metrics.json:
          cache: false
    plots:
      - plots/cm_validation.png:
          x: actual
          y: predicted
```

**Run:**
```bash
dvc repro train_rf
```

## 📖 API Reference

### DnsRfModel

```python
class DnsRfModel:
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 class_weight: Optional[str] = None,
                 random_state: int = 42,
                 n_jobs: int = -1) -> None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DnsRfModel"
    def predict(self, X: pd.DataFrame) -> np.ndarray
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray
    def evaluate(self, X: pd.DataFrame, y: pd.Series, 
                 set_name: str = "validation") -> dict
    def save(self, path: str) -> None
    @staticmethod
    def load(path: str) -> "DnsRfModel"
    def get_confusion_matrix(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray
    def get_feature_importance(self) -> Tuple[np.ndarray, int]
```

## ✅ Checklist

Before training in production:

- [ ] Data files exist and are readable
- [ ] Features have correct dtypes (float32 for continuous, int32 for categorical)
- [ ] Target column has binary values (0, 1)
- [ ] No NaN values in features or target
- [ ] Train/validation split strategy documented
- [ ] Hyperparameters validated (reasonable ranges)
- [ ] MLflow tracking URI configured
- [ ] Output directories exist or auto-created
- [ ] Logging configured appropriately
- [ ] Model save path is writable

## 🎓 Best Practices

1. **Reproducibility:**
   - Set `random_state` consistently
   - Document data source and preprocessing
   - Save config with each run

2. **Experiment Management:**
   - Use descriptive `run_name` for easy identification
   - Add meaningful tags for filtering
   - Compare runs in MLflow UI before deployment

3. **Model Monitoring:**
   - Track validation metrics across runs
   - Alert on metric degradation
   - Re-train regularly with new data

4. **Hyperparameter Tuning:**
   - Use grid search or Bayesian optimization
   - Log all runs for comparison
   - Consider computational cost vs. accuracy trade-off

## 📝 Next Steps

1. **Train baseline model:**
   ```bash
   python scripts/train/train_rf.py --config configs/train_rf.yaml
   ```

2. **View results:**
   ```bash
   mlflow ui
   ```

3. **Tune hyperparameters:**
   - Modify `configs/train_rf.yaml`
   - Re-run with different parameters
   - Compare in MLflow dashboard

4. **Deploy best model:**
   ```bash
   python scripts/deploy/package_model.py --run-id <best-run-id>
   ```

5. **Monitor in production:**
   - Set up model serving
   - Log predictions and metrics
   - Track model drift

## 📚 References

- MLflow Documentation: https://mlflow.org/docs
- Scikit-learn RandomForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- MLflow Model Registry: https://mlflow.org/docs/latest/model-registry.html
- Model Deployment: https://mlflow.org/docs/latest/models.html
