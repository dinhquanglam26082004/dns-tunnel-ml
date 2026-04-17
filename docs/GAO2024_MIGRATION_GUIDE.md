# Gao 2024 Dataset Integration & Migration Guide

## Overview

This guide describes the complete system for migrating from synthetic/test datasets to the **Gao et al. (2024) DNS Tunnel Detection Dataset**.

### Citation

```bibtex
@inproceedings{gao2024dns,
  title={DNS Tunnel Detection: A Systematic Classification and Benchmark Study},
  author={Gao, et al.},
  booktitle={Proceedings of YYYY},
  year={2024}
}
```

**Repository**: https://github.com/ggyggy666/DNS-Tunnel-Datasets

---

## Component Overview

### 1. **Cleanup System** (`scripts/utils/cleanup_old_data.py`)
Safe removal of old/temporary dataset files.

**Features**:
- 3 operation modes: `--dry-run`, `--execute --backup`, `--execute --no-backup`
- Automatic categorization: KEEP, REVIEW, DELETE
- Protected patterns (models/, configs/, scripts/, mlruns/)
- Detailed logging with backup to `.trash` directory

**Usage**:
```bash
# Preview what will be deleted
python scripts/utils/cleanup_old_data.py --dry-run

# Execute with safety backup
python scripts/utils/cleanup_old_data.py --execute --backup

# Permanent deletion (caution!)
python scripts/utils/cleanup_old_data.py --execute --no-backup
```

**Output**:
- Console + file logging to `logs/cleanup_YYYYMMDD_HHMMSS.log`
- Backup files moved to `data/.trash/`
- Summary report with freed space

---

### 2. **Integration System** (`scripts/data/integrate_gao2024.py`)
Download and parse Gao 2024 PCAP files to Parquet format.

**Features**:
- Automatic GitHub repo cloning or ZIP download
- Structure verification (benign/, known_tunnel/, unknown_tunnel/)
- Scapy-based PCAP parsing with chunked processing
- 6-feature extraction: qname, qtype, timestamp, label, tool, source_file
- Automatic label assignment: 0=benign, 1=malicious
- Zstd compression for optimal storage
- Tool diversity tracking

**Usage**:
```bash
# Download and parse
python scripts/data/integrate_gao2024.py --output data/raw

# Use existing dataset (skip download)
python scripts/data/integrate_gao2024.py --output data/raw --skip-download
```

**Output**:
- Parquet file: `data/raw/gao_dns_tunnel_2024_parsed.parquet`
- Statistics file: `data/raw/gao_dns_tunnel_2024_stats.json`
- Logging: `logs/integrate_gao2024.log`

**Extracted Features**:
```python
{
    "qname": "example.com",           # Domain name
    "qtype": 1,                       # DNS query type (A=1, AAAA=28, etc.)
    "timestamp": "2024-01-15 10:30:45",  # Query time
    "label": 0,                       # 0=benign, 1=malicious
    "tool": "iodine",                 # Tunneling tool or "benign"
    "source_file": "iodine_001.pcap"  # Source PCAP file
}
```

---

### 3. **Validation System** (`scripts/data/validate_gao2024.py`)
Comprehensive data quality checks.

**Validation Checks**:

| Check | Criteria | Status | Recommendation |
|-------|----------|--------|-----------------|
| **Completeness** | No NULL in qname, label, timestamp | PASS/FAIL | Critical |
| **Label Distribution** | 90/10 is acceptable, <10% = warning | PASS/WARN | Training strategy |
| **Tool Diversity** | ≥4 different tunneling tools | PASS/WARN | Dataset coverage |
| **Timestamp Consistency** | Valid timestamps, reasonable duration | PASS/WARN | Time range validation |
| **Domain Overlap** | Benign ↔ Malicious domain similarity | PASS/WARN | Data leakage check |
| **Feature Validity** | All required features present and valid | PASS/FAIL | Data schema |

**Usage**:
```bash
# Run validation
python scripts/data/validate_gao2024.py \
    --input data/raw/gao_dns_tunnel_2024_parsed.parquet \
    --output outputs/gao2024_validation_report.json
```

**Exit Codes**:
- `0` = All checks passed ✅
- `1` = Critical failure ❌
- `2` = Warnings found (data usable) ⚠

**Output**:
- JSON report: `outputs/gao2024_validation_report.json`
- Logging: `logs/validate_gao2024.log`

**Report Example**:
```json
{
  "overall_status": "PASS",
  "exit_code": 0,
  "checks": {
    "completeness": {
      "status": "PASS",
      "null_counts": {"qname": 0, "label": 0, "timestamp": 0}
    },
    "label_distribution": {
      "distribution": {0: 750000, 1: 250000},
      "ratios": {0: 75.0, 1: 25.0}
    },
    "tool_diversity": {
      "unique_count": 6,
      "tools": {"iodine": 150000, "dnscat2": 100000, ...}
    }
  },
  "recommendations": [...]
}
```

---

### 4. **Workflow Orchestration** (`scripts/workflows/upgrade_to_gao2024.ps1`)
PowerShell script that orchestrates the complete migration pipeline.

**Phases**:
1. **Cleanup** - Remove old temporary data with safety confirmation
2. **Integration** - Download and parse Gao 2024 dataset
3. **Validation** - Check data quality (exit on critical failures)
4. **Pipeline Rebuild** - Create train/val/test splits
5. **Training** - Retrain Random Forest model
6. **Evaluation** - Assess model performance

**Usage**:
```powershell
# Interactive workflow (prompts for confirmation)
.\scripts\workflows\upgrade_to_gao2024.ps1

# Dry-run (preview all operations)
.\scripts\workflows\upgrade_to_gao2024.ps1 -DryRun

# Skip cleanup phase
.\scripts\workflows\upgrade_to_gao2024.ps1 -SkipCleanup

# Skip training (only integration + validation)
.\scripts\workflows\upgrade_to_gao2024.ps1 -SkipTraining

# Combined options
.\scripts\workflows\upgrade_to_gao2024.ps1 -DryRun -Verbose
```

**Parameters**:
- `-DryRun`: Preview mode (no actual changes)
- `-SkipCleanup`: Skip data cleanup phase
- `-SkipDownload`: Use existing dataset directory
- `-SkipTraining`: Skip model retraining
- `-Verbose`: Detailed logging

**Output**:
- Timestamped console output with colored status
- Phase-by-phase progress tracking
- Final summary with output file locations

---

## Quick Start Guide

### Option 1: Full Automated Migration (Recommended)

```powershell
cd e:\dns-tunnel-ml
.\scripts\workflows\upgrade_to_gao2024.ps1
```

**What happens**:
1. Checks Python environment
2. Shows cleanup dry-run (prompts for confirmation)
3. Downloads Gao 2024 dataset from GitHub
4. Parses PCAP files to Parquet
5. Validates data quality
6. Rebuilds data pipeline
7. Trains new Random Forest model
8. Evaluates performance

**Expected duration**: 30-60 minutes (depends on download speed + dataset size)

---

### Option 2: Step-by-Step Manual Process

#### Step 1: Cleanup Old Data

```bash
# Preview
python scripts/utils/cleanup_old_data.py --dry-run

# Execute (confirm when prompted)
python scripts/utils/cleanup_old_data.py --execute --backup
```

#### Step 2: Download & Parse

```bash
python scripts/data/integrate_gao2024.py --output data/raw
```

#### Step 3: Validate

```bash
python scripts/data/validate_gao2024.py \
    --input data/raw/gao_dns_tunnel_2024_parsed.parquet \
    --output outputs/gao2024_validation_report.json
```

#### Step 4: Rebuild Pipeline

```bash
python scripts/data/build_pipeline.py \
    --input-dir data/raw \
    --output-dir data/splits_gao2024
```

#### Step 5: Train Model

```bash
python scripts/train/train_rf_1M.py \
    --config configs/train_rf_1M.yaml \
    --run-name "gao2024-production"
```

---

### Option 3: Dry-Run Preview

```powershell
# Preview all operations without making changes
.\scripts\workflows\upgrade_to_gao2024.ps1 -DryRun
```

---

## Dataset Structure

### Gao 2024 Repository Structure
```
DNS-Tunnel-Datasets/
├── benign/
│   ├── benign_001.pcap
│   ├── benign_002.pcap
│   └── ...
├── known_tunnel/
│   ├── iodine_001.pcap
│   ├── dnscat2_001.pcap
│   ├── dns2tcp_001.pcap
│   └── ...
└── unknown_tunnel/
    ├── unknown_001.pcap
    └── ...
```

### Integrated Dataset Structure
```
data/raw/
├── gao_dns_tunnel_2024_parsed.parquet    (main dataset)
├── gao_dns_tunnel_2024_stats.json        (statistics)
└── ... (other raw datasets)

data/splits_gao2024/
├── train.parquet
├── val.parquet
├── test.parquet
├── train_strat.parquet               (stratified version)
├── val_strat.parquet
├── test_strat.parquet
└── pipeline_metadata.json
```

---

## Important Notes

### Protected Files (Never Deleted)

The cleanup system **never** deletes:
- ✅ `models/` - Trained model files
- ✅ `scripts/` - Source code
- ✅ `configs/` - Configuration files
- ✅ `outputs/` - Generated outputs
- ✅ `.mlruns/` - MLflow tracking
- ✅ `*.pkl` - Serialized objects
- ✅ `*.yaml` - Configuration files
- ✅ `*.json` - Metadata files

### Dataset Characteristics

**Gao 2024 Dataset**:
- Complete real-world DNS traffic captures
- Multiple tunneling tools (iodine, dnscat2, dns2tcp, etc.)
- Benign traffic from legitimate domains
- Temporal coverage across multiple days
- High-quality labels and metadata

**Feature Space**:
```python
features = {
    "qname": str,              # Domain name
    "qtype": int,              # DNS query type
    "timestamp": datetime,     # Query timestamp
    "label": int,              # 0=benign, 1=malicious
    "tool": str,               # Tunneling tool name
    "source_file": str         # Source PCAP filename
}
```

### Compression & Storage

- **Format**: Apache Parquet with Zstd compression
- **Compression Ratio**: 8-10x (typical for DNS data)
- **Read Speed**: 150+ MB/s
- **Storage**: ~50-100 MB depending on dataset size

---

## Troubleshooting

### Issue: Git Clone Fails

**Symptoms**: "Git clone failed" message

**Solution**:
```bash
# Install Git if not available
# Or use fallback ZIP download (automatic)
```

### Issue: Scapy Import Error

**Symptoms**: "ImportError: No module named 'scapy'"

**Solution**:
```bash
pip install scapy==2.7.0
```

### Issue: Validation Warnings

**Symptoms**: Exit code 2 (warnings found)

**Solution**:
```bash
# Check the validation report
cat outputs/gao2024_validation_report.json

# Apply recommendations from the report
# Training still proceeds but may need adjustment
```

### Issue: Out of Memory During Parsing

**Symptoms**: MemoryError when processing large PCAP

**Solution**:
```bash
# Script already uses chunked processing
# If still issues: manually parse smaller subsets
python scripts/data/integrate_gao2024.py --output data/raw --skip-download
```

---

## Monitoring & Logging

### Log Files

- **Cleanup**: `logs/cleanup_YYYYMMDD_HHMMSS.log`
- **Integration**: `logs/integrate_gao2024.log`
- **Validation**: `logs/validate_gao2024.log`
- **Training**: `logs/train_rf_*.log`

### MLflow Tracking

All training runs are tracked in MLflow:
```bash
mlflow ui  # Start MLflow UI (http://localhost:5000)
```

### Output Reports

- **Validation Report**: `outputs/gao2024_validation_report.json`
- **Training Metrics**: `outputs/model_metrics_gao2024.json`
- **Statistics**: `data/raw/gao_dns_tunnel_2024_stats.json`

---

## Performance Benchmarks

### Integration Phase
- Download time: 2-5 minutes (network dependent)
- Parsing time: 10-30 minutes (CPU dependent)
- Total PCAP files: 50-200 (varies)
- Output size: 50-200 MB parquet

### Validation Phase
- Completion time: <1 minute
- Checks: 6 comprehensive validations

### Training Phase
- Training time: 15-45 minutes (CPU cores dependent)
- Model size: 50-100 MB
- Cross-validation: 5-fold

---

## Best Practices

### 1. **Always Run Validation**
```bash
# Never skip validation - it catches data quality issues early
python scripts/data/validate_gao2024.py --input data/raw/gao_dns_tunnel_2024_parsed.parquet
```

### 2. **Use Dry-Run First**
```bash
# Preview all operations before execution
.\scripts\workflows\upgrade_to_gao2024.ps1 -DryRun
```

### 3. **Backup Strategy**
```bash
# Cleanup uses backup by default
python scripts/utils/cleanup_old_data.py --execute --backup
# Old files moved to data/.trash, can be recovered if needed
```

### 4. **Monitor Training**
```bash
# Watch training progress in real-time
mlflow ui
# Then open http://localhost:5000 in browser
```

### 5. **Version Comparison**
Compare metrics before/after migration:
```bash
# Old model metrics
cat outputs/model_metrics_synthetic.json

# New model metrics
cat outputs/model_metrics_gao2024.json
```

---

## FAQ

**Q: Can I keep the old synthetic data?**
- A: Yes, use `--SkipCleanup` flag to preserve it.

**Q: How long does the full migration take?**
- A: 1-2 hours including download, parsing, validation, and training.

**Q: What if validation fails?**
- A: Fix the issues identified in the validation report or contact data maintainers.

**Q: Can I running training skip training and reuse old model?**
- A: Yes, use `--SkipTraining` flag if you only want to update the dataset.

**Q: How do I revert if something goes wrong?**
- A: Old files are backed up to `data/.trash` - restore with file explorer.

---

## Support & Citation

**Repository**: https://github.com/ggyggy666/DNS-Tunnel-Datasets

**Citation**:
```
Gao et al. (2024). DNS Tunnel Detection: A Systematic Classification and Benchmark Study.
```

**Contact**: See GitHub repository for maintainer contact information.

---

**Last Updated**: April 17, 2026  
**Version**: 1.0  
**Status**: Production Ready ✅
