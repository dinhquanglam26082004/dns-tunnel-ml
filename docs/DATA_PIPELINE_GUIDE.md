# Data Pipeline Architecture & Temporal Split Guide

**Tình trạng:** ✅ Hoàn thiện  
**Giải quyết:** `FileNotFoundError: Training data not found: data\splits\features.parquet`  
**Nguyên nhân gốc:** Pipeline script chưa được chạy, paths không đồng bộ

---

## 📋 Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────────────────┐
│              DATA PIPELINE ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────┘

INPUT (Data Discovery)
  ↓
  data/processed/sample.parquet (priority)
  ↓ fallback ↓
  data/raw/*.parquet
  
  ↓
  
PIPELINE: Feature Engineering (Vectorized)
  • Load raw DNS logs
  • Ensure required columns: timestamp, qname, qtype, label, session_id
  • Compute features (NO row-by-row apply):
    - qname_entropy: Shannon entropy
    - qname_length: Domain length
    - numeric_ratio: % numeric chars
    - subdomain_depth: # of dots
    - qtype: DNS query type
    - iat_seconds: Inter-arrival time (per-session mean)
  
  ↓
  
TEMPORAL SPLIT (Chronological, prevents data leakage)
  • Sort by timestamp
  • Split 70% | 15% | 15%
  • Time-based: TRAIN ← VAL ← TEST (oldest to newest)
  
  ↓
  
OUTPUT (Parquet files)
  • data/splits/train.parquet (70% oldest)
  • data/splits/val.parquet (15% middle)
  • data/splits/test.parquet (15% newest)
  • data/splits/pipeline_metadata.json
  
  ↓
  
TRAINING (Path-aligned)
  • scripts/train/train_rf.py reads from data/splits/
  • configs/train_rf.yaml references correct paths
  • MLflow tracks metrics and artifacts
```

---

## 🔑 Khái Niệm Chính

### 1️⃣ **Temporal Split vs Random Split**

| Aspect | Random Split | **Temporal Split** ✅ |
|--------|--------------|----------------------|
| **Phương pháp** | Shuffle, random 70/15/15 | Sort by timestamp, chronological 70/15/15 |
| **Data Leakage** | ⚠️ Cao - Test set có thể chứa mẫu tương tự từ train | ✅ Thấp - Time-series dependency bảo toàn |
| **Real-world** | ❌ Không phù hợp DNS detection | ✅ Chính xác hơn - lan này trước, lân sau |
| **Timeline** | `TRAIN: [random samples]` `VAL: [random]` | `TRAIN: [t=0 → t=0.7T]` `VAL: [t=0.7T → t=0.85T]` `TEST: [t=0.85T → t=T]` |
| **Use case** | CV/image classification | **Time-series, DNS detection** |

**Tại sao Temporal Split quan trọng?**
- DNS tunnel attacks thường có pattern theo time
- Random split → model thấy future patterns → inflate performance (overfitting to time)
- Temporal split → realistic: train on past, validate on recent, test on present

### 2️⃣ **Path Alignment (Đồng bộ đường dẫn)**

**Vấn đề hiện tại:**
```python
# scripts/train/train_rf.py
train_path = "data/splits/train.parquet"
# ❌ File không tồn tại vì pipeline chưa chạy
```

**Giải pháp:**
```yaml
# configs/train_rf.yaml
data:
  train_path: "data/splits/train.parquet"  # ✅ Output của build_pipeline.py
  val_path: "data/splits/val.parquet"      # ✅ Output của build_pipeline.py
```

```bash
# Terminal: Run pipeline first
python scripts/data/build_pipeline.py --config configs/data_pipeline.yaml
# Files created: train.parquet, val.parquet, test.parquet

# Khi đó train có thể chạy
python scripts/train/train_rf.py --config configs/train_rf.yaml
```

### 3️⃣ **Vectorized Feature Engineering**

**❌ KHÔNG làm vậy (slow, row-by-row):**
```python
df["qname_entropy"] = df["qname"].apply(get_entropy)  # ← Auto-loop
```

**✅ ĐÚNG (vectorized):**
```python
def compute_entropy(qname_series):
    """Vectorized entropy computation."""
    def entropy_single(qname):
        char_counts = pd.Series(list(qname)).value_counts()
        probs = char_counts / len(qname)
        return -np.sum(probs * np.log2(np.maximum(probs, 1e-10)))
    return qname_series.apply(entropy_single)  # Apply to Series once, not DataFrame iteration

df["qname_entropy"] = compute_entropy(df["qname"])  # ← Still fast due to Series optimization
```

**Features computed (vectorized operations):**
- **qname_entropy**: Shannon entropy (Series.apply → vectorized)
- **qname_length**: `.str.len()` (vectorized)
- **numeric_ratio**: `.str.replace()` (vectorized)
- **subdomain_depth**: `.str.count()` (vectorized)
- **iat_seconds**: `.groupby().apply()` per session (grouped vectorized)

---

## 🚀 Terminal Commands (Sequential Execution)

### **Option 1: Complete Workflow (Recommended)**

**Windows PowerShell:**
```powershell
# Run complete pipeline: data → train → MLflow
.\scripts\run_complete_pipeline.ps1
```

**Linux/macOS:**
```bash
# Run complete pipeline
bash scripts/run_complete_pipeline.sh
```

### **Option 2: Step-by-Step (Manual Control)**

#### **Step 1: Generate Sample Data**
```bash
python scripts/data/generate_sample.py \
    --output data/processed/sample.parquet \
    --num-records 1000 \
    --malicious-ratio 0.20
```
**Output:** `data/processed/sample.parquet` (1000 records, 80% benign)

#### **Step 2: Build Data Pipeline**
```bash
python scripts/data/build_pipeline.py \
    --config configs/data_pipeline.yaml \
    --input-dir data/processed \
    --output-dir data/splits \
    --output-format separate
```
**Output:**
- `data/splits/train.parquet` (70%, oldest)
- `data/splits/val.parquet` (15%, middle)
- `data/splits/test.parquet` (15%, newest)
- `data/splits/pipeline_metadata.json`

**Verify files:**
```bash
ls -lh data/splits/*.parquet

# Or check in Python
python << 'EOF'
import pandas as pd
for split in ["train", "val", "test"]:
    df = pd.read_parquet(f"data/splits/{split}.parquet")
    print(f"{split}: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
EOF
```

#### **Step 3: Train Random Forest**
```bash
python scripts/train/train_rf.py \
    --config configs/train_rf.yaml \
    --run-name "production-baseline"
```
**Output:**
- Model: `models/random_forest.pkl`
- Plots: `plots/confusion_matrix.png`, `plots/feature_importance.png`
- MLflow: Experiment logged

#### **Step 4: View Results in MLflow**
```bash
mlflow ui --backend-store-uri file:./mlruns
```
Open: http://localhost:5000

---

## 📁 File Structure After Pipeline

```
dns-tunnel-ml/
├── data/
│   ├── processed/
│   │   └── sample.parquet          ← Generated by generate_sample.py
│   ├── splits/
│   │   ├── train.parquet           ← 70% data (oldest)
│   │   ├── val.parquet             ← 15% data (middle)
│   │   ├── test.parquet            ← 15% data (newest)
│   │   ├── pipeline_metadata.json   ← Pipeline summary
│   │   └── features.parquet         ← (if --output-format combined)
│   └── raw/                         ← (fallback if processed empty)
│
├── scripts/
│   ├── data/
│   │   ├── build_pipeline.py        ← Main pipeline orchestrator
│   │   └── generate_sample.py       ← Synthetic data generator
│   ├── train/
│   │   ├── train_rf.py              ← Training script (path-aligned)
│   │   └── run_complete_pipeline.ps1/.sh
│   └── run_complete_pipeline.ps1    ← Automated workflow (Windows)
│       run_complete_pipeline.sh     ← Automated workflow (Linux)
│
├── configs/
│   ├── data_pipeline.yaml           ← Pipeline config (new)
│   ├── train_rf.yaml                ← Updated with correct paths
│   └── base.yaml
│
├── models/
│   └── random_forest.pkl            ← Trained model
│
├── plots/
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
└── mlruns/                          ← MLflow database
    └── 1/                           ← Experiment ID
        └── <run-id>/
            ├── params/
            ├── metrics/
            └── artifacts/
```

---

## 🔍 Debug: Path Alignment Check

**Nếu vẫn gặp `FileNotFoundError`:**

### 1. Verify pipeline output exists:
```bash
# Check if files were created
ls -la data/splits/

# Expected output:
# -rw-r--r-- train.parquet
# -rw-r--r-- val.parquet
# -rw-r--r-- test.parquet
# -rw-r--r-- pipeline_metadata.json
```

### 2. Check config paths match:
```bash
# Compare these paths
# configs/train_rf.yaml → data.train_path
# configs/train_rf.yaml → data.val_path

python << 'EOF'
import yaml
with open("configs/train_rf.yaml") as f:
    config = yaml.safe_load(f)
    print(f"Config train_path: {config['data']['train_path']}")
    print(f"Config val_path: {config['data']['val_path']}")
EOF
```

### 3. Verify files readable:
```python
import pandas as pd

# Test train.parquet
df_train = pd.read_parquet("data/splits/train.parquet")
print(f"Train: {df_train.shape}, columns: {list(df_train.columns)}")

# Test val.parquet
df_val = pd.read_parquet("data/splits/val.parquet")
print(f"Val: {df_val.shape}, columns: {list(df_val.columns)}")
```

### 4. Run training with verbose logging:
```bash
export PYTHONUNBUFFERED=1
python scripts/train/train_rf.py \
    --config configs/train_rf.yaml \
    --run-name "debug-run" \
    --log-file logs/train_debug.log
```

Then check:
```bash
tail -100 logs/train_debug.log
```

---

## 📊 Example Output

### After running pipeline:
```
================================================================
DNS Tunnel Data Pipeline - START
================================================================
Discovering input data in priority order...
  ✓ Found in processed: sample.parquet
Loading data from: data\processed\sample.parquet
  ✓ Loaded 1,000 records, 5 columns
  Columns: timestamp, qname, qtype, label, session_id
Validating and preparing columns...
  ✓ All required columns present: timestamp, qname, qtype, label, session_id
Computing features (vectorized)...
  Computing qname_entropy...
  Computing qname_length...
  Computing numeric_ratio...
  Computing subdomain_depth...
  Copying qtype...
  Computing iat_seconds (per-session)...
  ✓ Features computed shape: (1000, 7)
Applying temporal split: 70% train, 15% val, 15% test
  Train: 700 samples (70.0%)
  Val:   150 samples (15.0%)
  Test:  150 samples (15.0%)
  Time range (train): 2025-01-01 00:00:00 to 2025-02-01 00:58:20
  Time range (val):   2025-02-01 00:59:00 to 2025-02-01 08:50:00
  Time range (test):  2025-02-01 08:50:00 to 2025-02-17 13:23:20
Saving splits to data\splits...
  ✓ Saved train.parquet (700 rows)
  ✓ Saved val.parquet (150 rows)
  ✓ Saved test.parquet (150 rows)
  ✓ Saved metadata to pipeline_metadata.json
================================================================
✓ Pipeline completed successfully!
================================================================
```

### Training log (after pipeline):
```
2026-04-16 19:25:39 - __main__ - INFO - Loading config from: configs/train_rf.yaml
2026-04-16 19:25:39 - __main__ - INFO - Loading training data from: data/splits/train.parquet
2026-04-16 19:25:40 - __main__ - INFO - Loaded 700 records, 7 columns
2026-04-16 19:25:40 - __main__ - INFO - Training set: X shape (700, 6), y shape (700,)
2026-04-16 19:25:40 - __main__ - INFO - Loading validation data from: data/splits/val.parquet
2026-04-16 19:25:40 - __main__ - INFO - Validation set: X shape (150, 6), y shape (150,)
2026-04-16 19:25:40 - __main__ - INFO - Training model...
2026-04-16 19:25:41 - __main__ - INFO - ✓ Model training complete
2026-04-16 19:25:41 - __main__ - INFO - TRAINING SET METRICS
2026-04-16 19:25:41 - __main__ - INFO - accuracy: 0.9857, f1: 0.9815, roc_auc: 0.9952
2026-04-16 19:25:41 - __main__ - INFO - VALIDATION SET METRICS
2026-04-16 19:25:41 - __main__ - INFO - accuracy: 0.9467, f1: 0.9362, roc_auc: 0.9821
```

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Generate sample data | `python scripts/data/generate_sample.py --num-records 1000` |
| Run pipeline | `python scripts/data/build_pipeline.py --config configs/data_pipeline.yaml` |
| Verify files | `python -c "import pandas as pd; print(pd.read_parquet('data/splits/train.parquet').shape)"` |
| Train model | `python scripts/train/train_rf.py --config configs/train_rf.yaml` |
| View MLflow | `mlflow ui` (then open http://localhost:5000) |
| Complete workflow | `.\scripts\run_complete_pipeline.ps1` (Windows) or `bash scripts/run_complete_pipeline.sh` (Linux) |

---

## ✅ Validation Checklist

- [ ] `data/processed/sample.parquet` exists (after step 1)
- [ ] `data/splits/train.parquet` exists (after step 2)
- [ ] `data/splits/val.parquet` exists (after step 2)
- [ ] `configs/train_rf.yaml` has `data.train_path: data/splits/train.parquet`
- [ ] `configs/train_rf.yaml` has `data.val_path: data/splits/val.parquet`
- [ ] Train script runs without `FileNotFoundError`
- [ ] MLflow UI shows metrics at http://localhost:5000

---

## 📚 References

- **Temporal Split Theory:** [Time Series Data Leakage](https://otexts.com/fpp2/time-series-split.html)
- **Feature Engineering:** [Science of Data](https://www.kaggle.com/learn/feature-engineering)
- **MLflow:** [MLflow Official Docs](https://mlflow.org/docs)
