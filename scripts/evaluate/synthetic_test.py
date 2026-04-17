#!/usr/bin/env python3
"""
Synthetic DNS Query Interactive Testing.

This script manually creates specific raw DNS query strings and extracts features 
to test models behavior (Pure RF).
"""

import sys
import collections
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# Reconfigure stdout to avoid UnicodeEncodeError on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import warnings
warnings.filterwarnings("ignore")

# Define synthetic queries with their expected category
SYNTHETIC_DATA = [
    # NORMAL / BENIGN
    {"qname": "www.google.com", "category": "Normal"},
    {"qname": "api.github.com", "category": "Normal"},
    {"qname": "update.windows.microsoft.com", "category": "Normal"},
    
    # CDN / WILDCARD (Often false positives)
    {"qname": "server-192-0-2-1.cdn-edge.vid.cloudfront.net", "category": "CDN"},
    {"qname": "video-cache-abc-1234.s3.amazonaws.com", "category": "CDN"},
    
    # DNS TUNNELING (Base32/Base64/Hex - high entropy, long labels, few vowels)
    {"qname": "vvnzx94y3q9z8q83vvx38zvvnzx94y3q9z8q83vvx38z.tunnel.evil.com", "category": "Tunnel"},
    {"qname": "7h88dk98g70k0k08kk897h88dk98g70k0k08kk897h88dk98g70k0k08kk89.c2.anotherevil.net", "category": "Tunnel"},
    {"qname": "AAAABBBBCCCCDDDDEEEEFFFFAAAABBBBCCCCDDDDEEEEFFFF.base32.tunnel.org", "category": "Tunnel"},
    {"qname": "x02.a1bc.d2ef.g3hi.j4kl.m5no.p6qr.s7tu.v8wx.y9zz.sub.domain.malware.com", "category": "Tunnel"},
]

FEATURES = [
    "qname_entropy",
    "qname_length",
    "numeric_ratio",
    "subdomain_depth",
    "qtype",
    "max_label_len",
    "vowel_ratio",
    "unique_char_ratio",
]

VOWELS = set("aeiouAEIOU")

def compute_features_synthetic(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Compute the 8 core features expected by the models."""
    out = pd.DataFrame(index=df_raw.index)
    qname = df_raw["qname"].fillna("").astype(str)

    # 1. qname_entropy
    def entropy(q):
        if not q: return 0.0
        n = len(q)
        c = collections.Counter(q)
        p = np.fromiter((v / n for v in c.values()), dtype=np.float64)
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-10))))
    out["qname_entropy"] = qname.map(entropy)

    # 2. qname_length
    out["qname_length"] = qname.str.len()

    # 3. numeric_ratio
    num_len = qname.str.replace(r"\D", "", regex=True).str.len().fillna(0)
    out["numeric_ratio"] = num_len / out["qname_length"].clip(lower=1)

    # 4. subdomain_depth
    out["subdomain_depth"] = qname.str.count(r"\.")

    # 5. qtype
    out["qtype"] = 1  # 1 for A record by default for synthetic data

    # 6. max_label_len
    def max_label(q):
        parts = q.split(".")
        return max((len(p) for p in parts), default=0) if parts else 0
    out["max_label_len"] = qname.map(max_label)

    # 7. vowel_ratio
    def vowel_ratio(q):
        if not q: return 0.0
        alpha = sum(1 for c in q if c.isalpha())
        if alpha == 0: return 0.0
        return sum(1 for c in q if c in VOWELS) / alpha
    out["vowel_ratio"] = qname.map(vowel_ratio)

    # 8. unique_char_ratio
    def unique_char_ratio(q):
        if not q: return 0.0
        return len(set(q)) / len(q)
    out["unique_char_ratio"] = qname.map(unique_char_ratio)

    return out[FEATURES]


def load_model_from_file(path_str):
    """Safely handles DnsRfModel or Raw Model wrappers."""
    path = Path(path_str)
    if not path.exists():
        print(f"Error: Model not found at {path}")
        return None
    # Since models were natively saved with joblib.dump(self.model), joblib works
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None

def predict_class_name(prob_tunnel, cutoff=0.5):
    """Transforms numerical prob to visual tag"""
    if prob_tunnel >= cutoff:
        return f"TUNNEL ({(prob_tunnel*100):.1f}%)"
    else:
        return f"BENIGN ({(prob_tunnel*100):.1f}%)"

def main():
    print("="*80)
    print(" SYNTHETIC DATA EVALUATION: PURE RANDOM FOREST ")
    print("="*80)
    
    # 1. Provide initial DataFrame
    df_raw = pd.DataFrame(SYNTHETIC_DATA)
    print("\n[+] Extracting Features...")
    X_test = compute_features_synthetic(df_raw)
    
    # 2. Load models
    rf_path = "models/random_forest.pkl"
    
    print(f"[+] Loading RF Model: {rf_path}")
    rf_model = load_model_from_file(rf_path)
    
    if not rf_model:
        return
        
    print("\n" + "-"*85)
    print(f"{'QNAME (QUERY)':<45} | {'CATEGORY':<10} | {'PURE RF DIAGNOSIS':<22}")
    print("-" * 85)
    
    rf_probs = rf_model.predict_proba(X_test.values)[:, 1] if hasattr(rf_model, "predict_proba") else rf_model.predict(X_test.values)

    all_rf_correct = True

    for i in range(len(df_raw)):
        qname = df_raw.iloc[i]["qname"]
        cat = df_raw.iloc[i]["category"]
        
        # Display name truncating if too long
        display_qname = qname if len(qname) <= 42 else qname[:39] + "..."
        
        rf_p = rf_probs[i]
        
        rf_result = predict_class_name(rf_p)
        
        print(f"{display_qname:<45} | {cat:<10} | {rf_result:<22}")

    print("-" * 85)
    print("\n[INFO] Model analyzed the synthetic dataset.")
    print("Check if CDN/Normal are kept at VERY LOW Tunnel probabilities (< 5%), ")
    print("And if Base32 variations peak heavily into TUNNEL ranges (> 90%).\n")

if __name__ == "__main__":
    main()
