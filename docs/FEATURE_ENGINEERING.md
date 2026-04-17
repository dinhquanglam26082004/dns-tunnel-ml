# Feature Engineering & EDA Guide

## Overview

Complete feature engineering and exploratory data analysis pipeline for DNS tunnel detection, with separate paths for:
- **Random Forest models**: Tabular features from individual DNS queries
- **LSTM models**: Temporal sequences grouped by session

## Architecture

```
Raw DNS Data (data/processed/*.parquet)
        ↓
extract_dns_features()  ← src/features/extractor.py
        ↓
Features DataFrame (qname_entropy, qname_length, iat_seconds, ...)
        ↓
        ├─→ Random Forest training
        │   └─→ models/rf_model.pkl
        │
        └─→ build_lstm_sequences()  ← src/features/sequence_builder.py
            └─→ Padded sequences (N, 30, 6)
                └─→ LSTM training
                    └─→ models/lstm_model.h5
```

## Feature Set

### DNS-Specific Features (extracted per query)

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `qname_entropy` | float32 | [0, 8] | Shannon entropy of domain name characters |
| `qname_length` | int32 | [1, 255] | Character count of full domain name |
| `numeric_ratio` | float32 | [0, 1] | Proportion of numeric characters in qname |
| `subdomain_depth` | int32 | [1, ∞) | Number of labels (separated by dots) |
| `qtype` | int32 | [1, 255] | DNS query type (1=A, 28=AAAA, 5=CNAME, 33=SRV, etc.) |
| `iat_seconds` | float32 | [0, ∞) | Time since previous query in same session |
| `label` | int8 | {0, 1} | Target: 0=benign, 1=malicious DNS tunnel |

### Feature Engineering Details

#### Shannon Entropy (`qname_entropy`)
- Measures randomness/disorder in domain name
- Computed: Probability distribution of characters
- Tunnel domains often have higher entropy (random subdomains)
- Formula: $H = -\sum_{i} p_i \log_2(p_i)$

**Example:**
- "google.com" → entropy ≈ 3.8 (familiar patterns)
- "xdh9kdj.tunnel.bit" → entropy ≈ 6.2 (random characters)

#### Domain Length (`qname_length`)
- Total characters in FQDN
- Longer domains may indicate encoding/exfiltration
- Computed: `len(qname)`

#### Numeric Ratio (`numeric_ratio`)
- Proportion of numeric digits in domain
- Tunnel domains often use numbers for encoding
- Computed: `count("0"-"9") / len(qname)`

#### Subdomain Depth (`subdomain_depth`)
- Number of DNS labels (separated by ".")
- Excessive subdomains may indicate tunnel patterns
- Computed: `qname.count(".") + 1`

#### Inter-Arrival Time (`iat_seconds`)
- Time elapsed since previous query from same source to same domain
- **Key point**: Computed within session groups only
- Grouped by: `(src_ip, base_domain)` then sorted by timestamp
- First query in session: IAT = 0 (placeholder)

**IAT Computation:**
```python
# Group by session
grouped = df.groupby(['src_ip', 'base_domain'])
# Sort within each group by timestamp
df['iat_seconds'] = grouped['timestamp'].diff().fillna(0)
```

## Usage

### Quick Start

```bash
# 1. Activate environment
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate.bat         # Windows

# 2. Run feature extraction (Random Forest + LSTM sequences)
python scripts/features/extract.py \
    --input data/processed/clean.parquet \
    --output-features data/splits/features.parquet \
    --output-sequences data/splits/sequences/ \
    --seq-len 30 \
    --stride 15 \
    --save-scaler
```

### Output Files

```
data/splits/
├── features.parquet              # Tabular features (RF training)
├── sequences/
│   ├── X_sequences.npy          # Shape: (N, 30, 6)
│   ├── y_sequences.npy          # Shape: (N,)
│   └── scaler.pkl               # RobustScaler (IAT normalization)
```

### Python API

#### Extract Features for Random Forest

```python
from src.features.extractor import extract_dns_features
import pandas as pd

# Load raw data
df = pd.read_parquet('data/processed/clean.parquet')

# Extract features
features_df = extract_dns_features(df)

# Output columns: qname_entropy, qname_length, numeric_ratio, ...
print(features_df.shape)  # (N_samples, 7 features + metadata)
```

#### Build Sequences for LSTM

```python
from src.features.sequence_builder import build_lstm_sequences
import numpy as np

# features_df: output from extract_dns_features()
X, y, scaler = build_lstm_sequences(
    features_df,
    seq_len=30,      # queries per sequence
    stride=15,       # sliding window step
    fit_scaler=True  # fit RobustScaler on IAT
)

print(f"X shape: {X.shape}")  # (N_sequences, 30, 6)
print(f"y shape: {y.shape}")  # (N_sequences,)

# Save scaler for inference
import pickle
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

## Data Leakage Prevention ⚠️

### Critical Rules

1. **Scaler Fit Only on Training Data**
   - Never fit scaler on test/validation data
   - In `build_lstm_sequences`: Provide `fit_scaler=True` only for train set
   - For val/test: Pass existing scaler, never refit

   ```python
   # CORRECT: Fit scaler on train set
   X_train, y_train, scaler = build_lstm_sequences(
       features_train, fit_scaler=True  # FIT here
   )
   
   # CORRECT: Use fitted scaler on val/test
   X_val, y_val, _ = build_lstm_sequences(
       features_val, scaler=scaler, fit_scaler=False  # NO refit
   )
   ```

2. **No Temporal Lookahead**
   - Features computed from query content only (not future queries)
   - IAT uses only past queries in same session
   - Window labels: average within window boundaries only

3. **Session Boundaries**
   - Session = (src_ip, base_domain) pair
   - No information shared between sessions
   - Train/val/test splits should respect session boundaries

### Recommended Train/Val/Test Strategy

```python
# Group by source IP and time period
# Ensures clean separation no temporal leakage

df['time_bucket'] = pd.cut(df['timestamp'], bins=3, labels=['early', 'mid', 'late'])

# Split
train = df[df['time_bucket'] == 'early']      # First 33% of time
val = df[df['time_bucket'] == 'mid']          # Middle 33%
test = df[df['time_bucket'] == 'late']        # Last 33%

# Or: Split by src_ip for multi-source data
train = df[df['src_ip'].str.endswith('.1')]
val = df[df['src_ip'].str.endswith('.2')]
test = df[df['src_ip'].str.endswith('.3')]
```

## Exploratory Data Analysis

### Jupyter Notebook

Interactive EDA notebook: `notebooks/01_eda.ipynb`

**Sections:**
1. Load and profile data
2. Extract DNS features
3. Compute IAT statistics
4. Feature distributions (benign vs malicious)
5. Correlation analysis
6. Build LSTM sequences
7. Session length analysis
8. Data leakage prevention checks

**Run:**
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Key Insights to Look For

#### Feature Separability (Benign vs Malicious)
- Visualize: Violin plots of each feature by class
- Look for: Clear separation indicates good predictive power
- Entropy, length, numeric_ratio typically good discriminators

#### Class Imbalance
- Compute: `minority_class / majority_class`
- Action: If > 1:5, consider SMOTE or class weights
- Check: Post-windowing imbalance (may change due to majority voting)

#### IAT Patterns
- Benign: Usually consistent, longer intervals
- Malicious: Often bursty (rapid sequences), high variance
- Visualization: Box plot / violin plot by class

#### Session Coverage
- Sessions < 30 queries: Padded with zeros at start
- Sessions > 30: Sliding window extraction
- Report: % of data lost due to padding, mean session size

## Common Issues

### Issue: High proportion of NaN features
**Cause:** Data quality problems or missing qname/timestamp
**Solution:** 
```python
# Check with validate.py first
python scripts/data/validate.py
cat data/validation/report.json | jq '.summary'
```

### Issue: Scaler values don't make sense (scale=0)
**Cause:** IAT values in test set not matching train distribution
**Solution:**
```python
# Fit scaler on train set ONLY
# Don't refit on val/test sets
```

### Issue: Sequences show no class difference
**Cause:** Majority voting in windows (30-query window may mix classes)
**Solution:**
- Reduce `seq_len` (e.g., 15 instead of 30)
- Increase `stride` (larger gaps make windows more homogeneous)
- Check `y_sequences.mean()` - if ~0.5, windows are balanced

### Issue: "ModuleNotFoundError: No module named 'src.features'"
**Solution:**
```bash
# Run from project root directory
cd /path/to/dns-tunnel-ml
python scripts/features/extract.py ...

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## CLI Reference

### extract.py

```bash
python scripts/features/extract.py \
    --input <INPUT_PARQUET> \
    [--output-features <FEATURES_PARQUET>] \
    [--output-sequences <SEQUENCES_DIR>] \
    [--seq-len <SEQ_LEN>] \
    [--stride <STRIDE>] \
    [--save-scaler]

# Arguments:
#   --input              Path to input Parquet/CSV file (REQUIRED)
#   --output-features    Path to save tabular features (optional)
#   --output-sequences   Directory to save sequences (optional)
#   --seq-len            LSTM sequence length (default: 30)
#   --stride             Sliding window stride (default: 15)
#   --save-scaler        Save RobustScaler to pickle file

# Examples:
# Features only
python scripts/features/extract.py \
    --input data/processed/clean.parquet \
    --output-features data/splits/features.parquet

# Sequences only
python scripts/features/extract.py \
    --input data/processed/clean.parquet \
    --output-sequences data/splits/sequences/

# Both with custom parameters
python scripts/features/extract.py \
    --input data/processed/clean.parquet \
    --output-features data/splits/features.parquet \
    --output-sequences data/splits/sequences/ \
    --seq-len 20 \
    --stride 10 \
    --save-scaler
```

## Integration with DVC

Add to `dvc.yaml` to track feature extraction pipeline:

```yaml
stages:
  features:
    cmd: python scripts/features/extract.py --input data/processed/clean.parquet --output-features data/splits/features.parquet --output-sequences data/splits/sequences/ --save-scaler
    deps:
      - data/processed/clean.parquet
      - scripts/features/extract.py
      - src/features/
    outs:
      - data/splits/features.parquet
      - data/splits/sequences/X_sequences.npy
      - data/splits/sequences/y_sequences.npy
      - data/splits/sequences/scaler.pkl
    plots:
      - data/validation/report.json
```

**Run:**
```bash
dvc repro features
```

## Performance Optimization

### Vectorization Tips

1. **Use NumPy/Pandas, avoid row-by-row apply()**
   ```python
   # SLOW: apply()
   df['entropy'] = df['qname'].apply(compute_entropy)
   
   # FAST: vectorized (already done in extractor.py)
   ```

2. **Pre-allocate arrays for sequences**
   ```python
   # In sequence_builder.py: Arrays pre-allocated with np.array()
   # instead of list.append() + conversion
   ```

3. **Pre-fit scaler before transforming many sequences**
   ```python
   # Fit once
   scaler = RobustScaler()
   scaler.fit(iat_train)
   
   # Transform many batches without refitting
   for batch in batches:
       transformed = scaler.transform(batch)
   ```

### Benchmark (on 100K records)

| Operation | Time |
|-----------|------|
| Load CSV | ~2s |
| Extract features | ~1.5s |
| Build sequences (stride=15) | ~3s |
| Save outputs | ~1s |
| **Total** | **~7.5s** |

## References

- Shannon Entropy: https://en.wikipedia.org/wiki/Entropy_(information_theory)
- RobustScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
- Sliding Windows: https://en.wikipedia.org/wiki/Sliding_window
- DNS Protocol: RFC 1035 (https://tools.ietf.org/html/rfc1035)

## Next Steps

After feature engineering:

1. **Train Random Forest**
   ```bash
   python scripts/train/train_rf.py --features data/splits/features.parquet
   ```

2. **Train LSTM**
   ```bash
   python scripts/train/train_lstm.py --sequences data/splits/sequences/
   ```

3. **Model Evaluation**
   ```bash
   python scripts/eval/evaluate.py --predictions predictions.csv
   ```
