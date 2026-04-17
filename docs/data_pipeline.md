# Data Ingestion & Validation Pipeline

Complete setup for downloading and validating DNS tunnel detection datasets.

## Overview

This pipeline consists of two stages:

1. **Download** (`scripts/data/download_datasets.py`) - Fetch datasets from configured sources
2. **Validate** (`scripts/data/validate.py`) - Validate schema and data quality using Great Expectations

## Quick Start

### 1. Run Full Pipeline

```bash
# Activate virtual environment first
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate.bat  # Windows

# Run full DVC pipeline
dvc repro
```

### 2. Run Individual Scripts

```bash
# Download datasets
python scripts/data/download_datasets.py

# Validate datasets
python scripts/data/validate.py
```

## Setup & Installation

All dependencies are already configured in `pyproject.toml`:

```bash
# Install project with all dependencies (done in setup.sh/setup.bat)
pip install -e ".[dev,serving,tracking,validation]"
```

Key packages:
- `requests` - HTTP downloads with retry logic
- `great-expectations` - Data validation framework
- `dvc` - Experiment and data pipeline tracking
- `pandas` - Data manipulation and loading

## File Structure

```
dns-tunnel-ml/
├── scripts/data/
│   ├── download_datasets.py    # Download raw datasets
│   └── validate.py             # Validate data quality
├── data/
│   ├── raw/                    # Raw downloaded datasets
│   ├── validation/             # Validation reports
│   └── metadata/
│       └── dataset_card.json   # Dataset metadata template
├── dvc.yaml                    # DVC pipeline definition
└── great_expectations/         # GE configuration (auto-created)
```

## Scripts Details

### `download_datasets.py`

**Purpose**: Download DNS tunnel datasets from configured sources

**Features**:
- Configurable dataset sources in `DATASET_SOURCES` dict
- Automatic retry logic (3 attempts with 2-second delay)
- Supports extraction of `.zip` and `.tar.gz` files
- Checks if file already exists (skip if present)
- Detailed logging with download progress

**Configuration**:
```python
DATASET_SOURCES = {
    "gao-dns-tunnel-2024": "https://...",
    "palau-dnstunneldata-2020": "https://...",
}
```

**Usage**:
```bash
python scripts/data/download_datasets.py
```

**Output**:
- Downloaded files stored in `data/raw/`
- Console logs with success/failure status

---

### `validate.py`

**Purpose**: Validate datasets against schema and data quality rules

**Validation Rules**:
- ✓ `qname` column exists and is string type
- ✓ `label` values only contain 0 or 1
- ✓ `qname` missing values < 5%
- ✓ `qtype` values in range [1, 255]

**Features**:
- Great Expectations integration (Pandas engine)
- Supports CSV and Parquet formats
- Generates detailed JSON report
- Auto-creates Great Expectations project directory

**Usage**:
```bash
python scripts/data/validate.py
```

**Output**:
- Validation report: `data/validation/report.json`
- Contains:
  - Per-file validation results
  - Check status (passed/failed)
  - Data statistics (shape, columns)
  - Summary metrics

**Report Example**:
```json
{
  "timestamp": "2024-04-16T10:30:00.000000",
  "total_files_validated": 2,
  "results": [
    {
      "filename": "gao_2024.csv",
      "shape": {"rows": 50000, "columns": 9},
      "all_checks_passed": true,
      "checks": [...]
    }
  ],
  "summary": {
    "passed": 2,
    "failed": 0,
    "errors": 0
  }
}
```

---

## DVC Pipeline

### File: `dvc.yaml`

Defines the complete data ingestion pipeline for DVC.

**Stages**:

1. **download**
   - Input: None
   - Output: `data/raw/` directory
   - Command: `python scripts/data/download_datasets.py`

2. **validate**
   - Input: `data/raw/` directory
   - Output: `data/validation/report.json`
   - Command: `python scripts/data/validate.py`

### DVC Commands

```bash
# Show pipeline DAG
dvc dag

# Run full pipeline
dvc repro

# Run specific stage
dvc repro validate

# Check pipeline status
dvc status

# View pipeline results
dvc metrics show data/validation/report.json
```

### DVC Tracking

Pipeline outputs are automatically tracked:

```bash
# Add outputs to git
git add dvc.yaml .gitignore

# Commit pipeline definition
git commit -m "Add data ingestion pipeline"

# Push data to DVC remote (optional)
dvc push
```

---

## Dataset Metadata

### File: `data/metadata/dataset_card.json`

Template for documenting dataset information. Fill in these sections:

- **dataset_info**: Name, version, source, collection date, license
- **data_collection**: Capture method, environment, protocols, anonymization
- **labeling**: Labeling method, label distribution, inter-annotator agreement
- **features**: Feature descriptions with types, ranges, examples
- **dataset_statistics**: Size, sample count, missing values
- **known_limitations**: Important caveats and limitations
- **remarks**: Preprocessing steps, data quality issues, recommendations

**Example**:
```json
{
  "dataset_info": {
    "dataset_name": "Gao DNS Tunnel 2024",
    "version": "1.0",
    "collected_date": "2024-01-01",
    "source_url": "https://github.com/gao/dns-tunnel-ml"
  },
  "features": {
    "feature_names": ["qname", "qtype", "label", "entropy", ...]
  }
}
```

---

## Troubleshooting

### Download fails: Connection timeout
```bash
# Manually set longer timeout (edit script)
REQUEST_TIMEOUT = 60  # seconds
MAX_RETRIES = 5

# Or change dataset source URL if mirror is unavailable
DATASET_SOURCES["dataset-name"] = "https://alternative-mirror-url"
```

### "Module not found: great_expectations"
```bash
# Ensure dev dependencies installed
pip install -e ".[validation]"
```

### Validation fails: Missing columns
- Check that CSV/Parquet files have required columns: `qname`, `label`, `qtype`
- Add custom checks to `create_expectations_suite()` in `validate.py`

### DVC errors
```bash
# Initialize DVC if not done
dvc init

# Check DVC status
dvc status

# Reset pipeline cache
dvc repro --force
```

---

## Integration with ML Pipeline

After validation:

```bash
# Check validation report
cat data/validation/report.json | jq '.summary'

# Next: Feature engineering
python scripts/data/feature_engineering.py
```

---

## Best Practices

1. **Always run validation before training**
   ```bash
   dvc repro
   ```

2. **Version data using DVC**
   ```bash
   git add data/metadata/dataset_card.json
   dvc add data/raw/
   ```

3. **Monitor validation reports**
   ```bash
   # Track validation metrics over time
   dvc metrics logs data/validation/report.json
   ```

4. **Update dataset sources as needed**
   ```python
   # In download_datasets.py
   DATASET_SOURCES: Dict[str, str] = {
       "new-dataset": "https://...",
   }
   ```

---

## References

- DVC Documentation: https://dvc.org/doc
- Great Expectations: https://docs.greatexpectations.io/
- DNS Tunnel References:
  - Gao et al. 2024: DNS Tunnel Detection via ML
  - Palau et al. 2020: DNS-based Covert Channel Analysis
  - CTU-13 Dataset: https://www.stratosphereips.org/datasets-ctu13
