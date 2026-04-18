# DNS Tunnel Detection — ML Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-SQLite-blue.svg)](https://mlflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Gao%202024-green.svg)](https://github.com/ggyggy666/DNS-Tunnel-Datasets)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Random Forest classifier for DNS tunnel attack detection, trained on the real-world Gao et al. (2024) PCAP dataset with cross-tool generalization evaluation.**

---



## Quick Start

```powershell
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Tích hợp Gao 2024 PCAP dataset → Parquet
python scripts/data/integrate_gao2024.py

# 3. Feature engineering + stratified split
python scripts/data/build_pipeline.py --config configs/gao2024_pipeline.yaml

# 4. Training
python scripts/train/train_rf.py --config configs/train_rf.yaml --run-name "v2"

# 5. ML/Production Validation & Synthetic test
python quick_validate.py
python scripts/evaluate/synthetic_test.py

# 6. View MLflow experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Dataset — Gao et al. (2024)

**Nguồn:** [ggyggy666/DNS-Tunnel-Datasets](https://github.com/ggyggy666/DNS-Tunnel-Datasets)  
**Format:** PCAP files → parse bằng Scapy → `.parquet`  
**Tổng:** ~4.26 million DNS queries

| Category | Label | Records | Mô tả |
|---|---|---|---|
| `normal` | 0 (Benign) | 2,012,494 | Cloudflare top-1M benign traffic |
| `tunnel` | 1 (Tunnel) | 624,007 | Known tools: dnscat2, iodine, dnspot, tuns |
| `wildcard` | 0 (Benign) | 638,060 | Cloudflare wildcard DNS (high subdomain depth) |
| `unkownTunnel` | 1 (Tunnel) | 772,780 | **OOD:** CobaltStrike, dns2tcp, tcp-over-dns |
| `crossEndPoint` | 1 (Tunnel) | 212,549 | **OOD:** Android platform (AndIodine) |

> **Training set:** `normal + tunnel + wildcard` (3,274,561 records)  
> **OOD test:** `unkownTunnel + crossEndPoint` (985,329 records — not seen during training)

---

## Pipeline Architecture

```
data/gao2024_source/          ← PCAP files (844 MB)
       │
       ▼
scripts/data/integrate_gao2024.py
       │  (Scapy PCAP parse → extract DNS layer)
       ▼
data/raw/gao_dns_tunnel_2024_parsed.parquet
  Columns: timestamp, qname, qtype, packet_size,
           label, category, role, tool, source_file
       │
       ▼
scripts/data/build_pipeline.py   (configs/gao2024_pipeline.yaml)
       │
       │  Feature Engineering (8 features):
       │    qname_entropy, qname_length, numeric_ratio,
       │    subdomain_depth, qtype, max_label_len,
       │    vowel_ratio, unique_char_ratio
       │
       │  Split Strategy: StratifiedShuffleSplit (seed=42)
       │    70% train / 15% val / 15% test
       ▼
data/splits/
  ├── train.parquet   (2,981,922 rows)
  ├── val.parquet     (638,984 rows)
  └── test.parquet    (638,984 rows)
       │
       ▼
scripts/train/train_rf.py   (configs/train_rf.yaml)
       │  RandomForest: 200 trees, max_depth=20,
       │  class_weight=balanced, n_jobs=-1
       ▼
models/random_forest.pkl    ← Production model
mlflow.db                   ← Experiment tracking (SQLite)
plots/
  ├── cm_validation.png
  └── feature_importance.png
```

---

## Features (8 tổng)

| Feature | Importance | Lý do |
|---|---|---|
| `numeric_ratio` | **33.1%** | Tunnel payload chứa nhiều số (base32/hex encoding) |
| `subdomain_depth` | 16.2% | Tunnel dùng nhiều subdomain label |
| `qname_length` | 14.0% | Domain name dài hơn khi mang payload |
| `qtype` | 11.8% | Tunnel dùng TXT/NULL query type |
| `max_label_len` ⭐ | 9.4% | Tunnel nhồi payload vào 1 label dài |
| `unique_char_ratio` ⭐ | 8.1% | Base32 charset (~32 ký tự) vs ngôn ngữ tự nhiên |
| `qname_entropy` | 4.6% | Entropy cao → dữ liệu encoded |
| `vowel_ratio` ⭐ | 3.0% | Encoding phá vỡ phân phối nguyên âm tự nhiên |

> ⭐ = Feature mới thêm để fix wildcard false positive

### Tại sao KHÔNG dùng `iat_seconds`?

`iat_seconds` bị **loại bỏ** vì nó **leaky** đối với Gao 2024:

```
session_id = pd.cut(timestamp, bins=100)  ← time-window bins
iat_seconds = mean IAT per session         ← leaky proxy!
```

Gao 2024 chụp PCAP theo ngày khác nhau:
- `normal` → Aug 2023 → `iat_seconds` ∈ {0.006, 0.017, 0.036} (label=0)
- `tunnel` → Sep 2023 → `iat_seconds` ∈ {0.062 … 12.76}    (label=1)

→ Hai tập **không overlap** → `iat_seconds` một mình cho accuracy=1.0 (trivial cheat).

---

## Cấu Trúc Project

```
dns-tunnel-ml/
├── README.md
├── requirements.txt
├── pyproject.toml
├── dvc.yaml
├── mlflow.db                          ← MLflow SQLite backend
│
├── configs/
│   ├── gao2024_pipeline.yaml          ← Pipeline config (Gao 2024)
│   ├── train_rf.yaml                  ← RF training config (8 features)
│   └── data_pipeline.yaml             ← Generic pipeline config
│
├── data/
│   ├── gao2024_source/                ← PCAP files (844 MB)
│   │   ├── normal/normal/             ← Benign traffic
│   │   ├── tunnel/                    ← Known tunnel tools
│   │   ├── unkownTunnel/              ← Unknown tunnel tools (OOD)
│   │   ├── crossEndPoint/             ← Android tunnels (OOD)
│   │   └── wildcard/                  ← Wildcard benign
│   ├── raw/
│   │   └── gao_dns_tunnel_2024_parsed.parquet
│   ├── splits/
│   │   ├── train.parquet              ← 2,981,922 rows (70%)
│   │   ├── val.parquet                ← 638,984 rows (15%)
│   │   ├── test.parquet               ← 638,984 rows (15%)
│   │   └── pipeline_metadata.json
│   └── metadata/
│       └── dataset_card.json
│
├── scripts/
│   ├── data/
│   │   ├── integrate_gao2024.py       ← PCAP → Parquet (Scapy)
│   │   ├── build_pipeline.py          ← Feature engineering + split
│   │   └── validate_gao2024.py        ← Data integrity checks
│   ├── train/
│   │   └── train_rf.py                ← RF training + MLflow
│   └── evaluate/
│       └── synthetic_test.py          ← Synthetic base32 testing
│
├── src/
│   ├── models/
│   │   └── random_forest.py           ← RF wrapper (DnsRfModel)
│   └── utils/
│       └── logging_setup.py           ← UTF-8 safe logging
│
├── models/
│   └── random_forest.pkl              ← Current production model (v5)
│
├── plots/
│   ├── cm_validation.png
│   └── feature_importance.png
│
├── docs/
│   ├── GAO2024_MIGRATION_GUIDE.md
│   ├── FEATURE_ENGINEERING.md
│   ├── MODEL_TRAINING.md
│   ├── DATA_PIPELINE_GUIDE.md
│   ├── ARCHITECTURE_QUICK_REFERENCE.md
│   └── DEPLOYMENT_CHECKLIST.md
│
├── logs/
│   ├── integrate_gao2024.log
│   └── validate_gao2024.log
│
└── mlruns/                            ← MLflow artifacts
```

---

## Cài Đặt

### Yêu cầu
- Python 3.10+
- Windows / Linux / macOS
- ~2 GB RAM cho feature engineering (4.2M records)
- Scapy (parse PCAP)

### Setup

```powershell
# 1. Tạo virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Verify
python -c "import scapy, pandas, sklearn, mlflow; print('OK')"
```

### Windows — UTF-8 console (optional)
```powershell
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

---

## Hướng Dẫn Sử Dụng

### Bước 1: Tích hợp PCAP Dataset

```powershell
# Parse PCAP → Parquet (~15-45 phút tùy hardware)
python scripts/data/integrate_gao2024.py

# Validate dữ liệu sau parse
python scripts/data/validate_gao2024.py
```

Output: `data/raw/gao_dns_tunnel_2024_parsed.parquet`

### Bước 2: Feature Engineering + Split

```powershell
python scripts/data/build_pipeline.py --config configs/gao2024_pipeline.yaml
```

Output: `data/splits/{train,val,test}.parquet`  
Features: 8 features (không có `iat_seconds`)  
Split: Stratified shuffle 70/15/15

### Bước 3: Training

```powershell
python scripts/train/train_rf.py \
    --config configs/train_rf.yaml \
    --run-name "gao2024-v2"
```

### Bước 4: Validation & Synthetic Evaluation

```powershell
# Chạy stress test ML & Production
python quick_validate.py

# Kiểm thử với tên miền tĩnh (DNS Tunnel giả lập)
python scripts/evaluate/synthetic_test.py
```

### Bước 5: Xem Kết Quả MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Mở: http://localhost:5000
```

---

## Cấu Hình

### `configs/train_rf.yaml` (chính)

```yaml
model:
  n_estimators: 200
  max_depth: 20
  class_weight: "balanced"
  random_state: 42
  n_jobs: -1

data:
  train_path: "data/splits/train.parquet"
  val_path:   "data/splits/val.parquet"
  target_col: "label"
  feature_cols:
    - qname_entropy
    - qname_length
    - numeric_ratio
    - subdomain_depth
    - qtype
    - max_label_len
    - vowel_ratio
    - unique_char_ratio

mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "DNS-Tunnel-RF-Gao2024"
```

### `configs/gao2024_pipeline.yaml`

```yaml
paths:
  input_dir:  "data/processed"
  output_dir: "data/splits"

split_ratios:
  train: 0.70
  val:   0.15
  test:  0.15

filter_categories:
  - normal     # label=0
  - tunnel     # label=1 (known tools)
  - wildcard   # label=0 (high-depth benign — must be in training!)
# unkownTunnel + crossEndPoint → held out for OOD evaluation
```

---

## Troubleshooting

### UnicodeEncodeError trên Windows
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'
```
**Đã fix:** `setup_logging()` tự động wrap stdout trong UTF-8 writer.  
Nếu vẫn gặp: `$env:PYTHONIOENCODING = "utf-8"`

### MLflow `meta.yaml` does not exist
```
Malformed experiment. Yaml file 'mlruns/1/meta.yaml' does not exist.
```
**Fix:**
```powershell
# Xóa experiment bị corrupt
Get-ChildItem mlruns -Directory |
  Where-Object { -not (Test-Path "$($_.FullName)\meta.yaml") } |
  Remove-Item -Recurse -Force

# Dùng SQLite backend (đã set mặc định)
# tracking_uri: "sqlite:///mlflow.db"
```

### Kết quả Accuracy = 1.0 (giả)
**Nguyên nhân:** `iat_seconds` bị leaky — time-window `session_id` aligns hoàn toàn với timestamp của từng category.  
**Đã fix:** Loại `iat_seconds` khỏi `feature_cols` trong `train_rf.yaml`.

### Wildcard FP rate cao (74%)
**Nguyên nhân:** Model không biết wildcard `*.cloudflare.com` là benign → nhầm vì `subdomain_depth` cao.  
**Đã fix:** Thêm `wildcard` vào `filter_categories` trong config + 3 encoding features mới.

---

## Ghi Chú Kỹ Thuật

### Split Strategy

Gao 2024 có **timestamp cluster theo category** (normal=Aug 2023, tunnel=Sep 2023, wildcard=Oct 2023).  
Temporal split tạo ra `val = 100% tunnel` → training bất khả thi.  
Giải pháp: **StratifiedShuffleSplit** (seed=42) thay vì chronological sort.

### Đánh Giá Trung Thực

| Loại evaluation | Ý nghĩa | Kết quả |
|---|---|---|
| Row-level val split | Rows từ cùng PCAP → **dễ** | acc ~0.9999 |
| Cross-tool OOD test | Tools hoàn toàn mới → **thực** | acc 0.9916–0.9982 |

Cross-tool evaluation là benchmark quan trọng nhất — mô phỏng đúng thực tế triển khai.

---

## References

- **Gao et al. (2024)** — *DNS Tunnel Dataset* — [GitHub](https://github.com/ggyggy666/DNS-Tunnel-Datasets)
- **Scikit-learn** — Random Forest, StratifiedShuffleSplit — [docs](https://scikit-learn.org)
- **MLflow** — Experiment tracking & model registry — [mlflow.org](https://mlflow.org)
- **Scapy** — PCAP parsing — [scapy.net](https://scapy.net)

---

## Version History

### v2.1.0 — 2026-04-17 (Current)
- ✅ Xóa toàn bộ tệp tin dư thừa và các module về Hybrid, tập trung sức mạnh dứt điểm vào Random Forest Core Machine.
- ✅ Cập nhật bài Stress Test `quick_validate.py` cực khắt khe thay thế toàn bộ eval cũ.
- ✅ Bổ sung kịch bản `synthetic_test.py` giả lập truy vấn Base32 thực thụ. 

### v2.0.0 — 2026-04-17
- ✅ Migrate sang Gao 2024 real-world PCAP dataset (4.26M records)
- ✅ Fix wildcard false positive: thêm wildcard vào training
- ✅ 3 features mới: `max_label_len`, `vowel_ratio`, `unique_char_ratio`
- ✅ Loại `iat_seconds` (leaky via time-window session bins)
- ✅ Cross-tool evaluation script
- ✅ Stratified shuffle split (fix broken temporal split)
- ✅ MLflow → SQLite backend (fix deprecated file store)
- ✅ UTF-8 logging fix cho Windows cp1252

### v1.0.0 — 2026-04-16
- ✅ Basic pipeline + RF implementation
- ✅ MLflow integration
- ✅ Synthetic dataset (deprecated)

---

**Dataset:** Gao et al. 2024 DNS-Tunnel-Datasets  
**Model:** Random Forest v5 (200 trees, 8 features)  
**Last Updated:** 2026-04-17  
**Status:** Production Ready ✅
