# Gao 2024 Integration System - Quick Reference

## Files in This System

### 1. Python Scripts

#### `scripts/utils/cleanup_old_data.py`
- **Purpose**: Safe cleanup of old/temporary dataset files
- **Modes**: `--dry-run`, `--execute --backup`, `--execute --no-backup`
- **Output**: Timestamped log to `logs/cleanup_*.log`
- **Safety**: Protected KEEP patterns, backup to `.trash`

#### `scripts/data/integrate_gao2024.py`
- **Purpose**: Download and parse Gao 2024 PCAP to Parquet
- **Input**: GitHub repository or existing PCAP files
- **Output**: `data/raw/gao_dns_tunnel_2024_parsed.parquet`
- **Features**: 6-field DNS records, tool tracking, zstd compression

#### `scripts/data/validate_gao2024.py`
- **Purpose**: Comprehensive data quality validation
- **Checks**: 6 validation checks (completeness, distribution, diversity, overlap, etc.)
- **Output**: JSON report at `outputs/gao2024_validation_report.json`
- **Exit codes**: 0=pass, 1=critical failure, 2=warnings

### 2. PowerShell Workflow

#### `scripts/workflows/upgrade_to_gao2024.ps1`
- **Purpose**: Orchestrate complete migration pipeline
- **Phases**: 6 phases (cleanup → integration → validation → rebuild → train → evaluate)
- **Parameters**: 
  - `-DryRun`: Preview mode
  - `-SkipCleanup`, `-SkipDownload`, `-SkipTraining`: Skip phases
  - `-Verbose`: Detailed logging

### 3. Documentation

#### `docs/GAO2024_MIGRATION_GUIDE.md`
- Complete migration guide with all details
- Step-by-step instructions
- Troubleshooting section
- Best practices and FAQ

---

## Quick Commands

### Option A: Fully Automated (Recommended)
```powershell
.\scripts\workflows\upgrade_to_gao2024.ps1
```

### Option B: Step-by-Step
```bash
# Step 1: Cleanup old data
python scripts/utils/cleanup_old_data.py --execute --backup

# Step 2: Download & parse Gao 2024
python scripts/data/integrate_gao2024.py --output data/raw

# Step 3: Validate
python scripts/data/validate_gao2024.py \
    --input data/raw/gao_dns_tunnel_2024_parsed.parquet \
    --output outputs/gao2024_validation_report.json

# Step 4: Build pipeline
python scripts/data/build_pipeline.py \
    --input-dir data/raw \
    --output-dir data/splits_gao2024

# Step 5: Train model
python scripts/train/train_rf_1M.py \
    --config configs/train_rf_1M.yaml \
    --run-name "gao2024-production"
```

### Option C: Preview Mode
```powershell
.\scripts\workflows\upgrade_to_gao2024.ps1 -DryRun
```

---

## Output Files

| File | Purpose |
|------|---------|
| `data/raw/gao_dns_tunnel_2024_parsed.parquet` | Main dataset (Parquet) |
| `data/raw/gao_dns_tunnel_2024_stats.json` | Dataset statistics |
| `outputs/gao2024_validation_report.json` | Validation results |
| `data/splits_gao2024/train/val/test.parquet` | Data splits |
| `logs/cleanup_*.log` | Cleanup log |
| `logs/integrate_gao2024.log` | Integration log |
| `logs/validate_gao2024.log` | Validation log |

---

## Key Features

✅ **Safe Cleanup**: Protected patterns, backup to `.trash`
✅ **Automatic Download**: Git or ZIP fallback from GitHub
✅ **PCAP Parsing**: Scapy-based with chunked processing
✅ **Quality Validation**: 6-point check with detailed report
✅ **Zstd Compression**: 8-10x compression ratio, fast read speeds
✅ **Complete Logging**: Timestamped logs for all operations
✅ **Dry-Run Support**: Preview all changes before execution
✅ **Error Handling**: Graceful failure with recovery options

---

## Dataset Info

**Repository**: https://github.com/ggyggy666/DNS-Tunnel-Datasets

**Citation**: Gao et al. (2024)

**Structure**:
- `benign/` - Legitimate DNS traffic
- `known_tunnel/` - Known tunneling tools
- `unknown_tunnel/` - Unknown/novel tunneling tools

**Tools Covered**: iodine, dnscat2, dns2tcp, ptunnel, tuns, and more

---

## Support

See `docs/GAO2024_MIGRATION_GUIDE.md` for detailed documentation, troubleshooting, and FAQ.
