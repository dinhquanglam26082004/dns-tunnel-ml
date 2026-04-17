"""
DNS Feature Extraction Module.

Vectorized computation of DNS-specific features for Random Forest models.
Avoids data leakage by computing features per sample independently.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy

logger = logging.getLogger(__name__)


def compute_entropy(domain: str) -> float:
    """
    Compute Shannon entropy of a domain name (characters only).

    Args:
        domain: Domain name string (e.g., "example.com")

    Returns:
        Shannon entropy value (0-8 for ASCII). NaN if domain is empty.
    """
    if not domain or pd.isna(domain):
        return np.nan

    # Remove dots, compute character entropy
    chars = domain.replace(".", "").lower()
    if not chars:
        return np.nan

    # Count character frequencies
    unique_chars, counts = np.unique(list(chars), return_counts=True)
    prob = counts / len(chars)
    return entropy(prob, base=2)


def compute_subdomain_depth(domain: str) -> int:
    """
    Count number of subdomain levels.

    Args:
        domain: Domain name string

    Returns:
        Number of dots + 1. NaN if domain is empty.
    """
    if not domain or pd.isna(domain):
        return np.nan
    return domain.count(".") + 1


def compute_numeric_ratio(domain: str) -> float:
    """
    Compute ratio of numeric characters in domain name.

    Args:
        domain: Domain name string

    Returns:
        Ratio of digits / total characters (0-1). NaN if domain is empty.
    """
    if not domain or pd.isna(domain):
        return np.nan

    domain_str = str(domain).replace(".", "")
    if not domain_str:
        return np.nan

    num_digits = sum(c.isdigit() for c in domain_str)
    return num_digits / len(domain_str)


def extract_dns_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract DNS-specific features from raw DNS query data.

    Performs vectorized computation of features without row-by-row iteration.
    Computes Inter-Arrival Time (IAT) within session groups (src_ip + base_domain).

    Args:
        df: Input DataFrame with columns:
            - qname (str): Full domain name
            - qtype (int): DNS query type (1-255)
            - label (int): 0=benign, 1=malicious
            - timestamp (float/int): Unix timestamp or relative time
            - src_ip (str): Source IP address

    Returns:
        DataFrame with engineered features and cleaned data.
        Columns include: qname_entropy, qname_length, numeric_ratio,
                        subdomain_depth, qtype, label, iat_seconds, src_ip, base_domain

    Drops:
        - Rows with missing qname or label
        - Rows with missing critical numeric features after computation
    """
    logger.info(f"Extracting features from {len(df)} records")

    # Create working copy
    df_feat = df.copy()

    # Required columns check
    required = {"qname", "label"}
    optional = {"qtype", "timestamp", "src_ip"}
    missing = required - set(df_feat.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Handle optional columns
    if "qtype" not in df_feat.columns:
        df_feat["qtype"] = 1  # Default to A record
    if "src_ip" not in df_feat.columns:
        df_feat["src_ip"] = "0.0.0.0"

    # Extract base domain (e.g., "example.com" from "sub.example.com")
    df_feat["base_domain"] = df_feat["qname"].str.split(".").str[-2:].str.join(".")

    # 1. Entropy of domain name (vectorized)
    logger.debug("Computing Shannon entropy...")
    df_feat["qname_entropy"] = df_feat["qname"].apply(compute_entropy)

    # 2. Domain length (vectorized)
    logger.debug("Computing domain length...")
    df_feat["qname_length"] = df_feat["qname"].str.len()

    # 3. Numeric character ratio (vectorized)
    logger.debug("Computing numeric ratio...")
    df_feat["numeric_ratio"] = df_feat["qname"].apply(compute_numeric_ratio)

    # 4. Subdomain depth (vectorized)
    logger.debug("Computing subdomain depth...")
    df_feat["subdomain_depth"] = df_feat["qname"].apply(compute_subdomain_depth)

    # 5. Inter-Arrival Time (IAT) within session groups
    if "timestamp" in df_feat.columns:
        logger.debug("Computing Inter-Arrival Time (IAT)...")
        # Sort by timestamp for IAT computation
        df_feat = df_feat.sort_values("timestamp").reset_index(drop=True)

        # Group by session (src_ip + base_domain) and compute time differences
        df_feat["iat_seconds"] = df_feat.groupby(["src_ip", "base_domain"], sort=False)[
            "timestamp"
        ].diff()

        # First query in each group has NaN IAT → set to 0
        df_feat["iat_seconds"] = df_feat["iat_seconds"].fillna(0)
    else:
        logger.warning("No 'timestamp' column; skipping IAT computation")
        df_feat["iat_seconds"] = 0

    # 6. Keep qtype as-is (already numeric)
    # Ensure qtype is integer
    df_feat["qtype"] = df_feat["qtype"].astype(np.int32)

    # 7. Ensure label is binary (0 or 1)
    df_feat["label"] = df_feat["label"].astype(np.int8)

    # Select feature columns
    feature_cols = [
        "qname_entropy",
        "qname_length",
        "numeric_ratio",
        "subdomain_depth",
        "qtype",
        "label",
        "iat_seconds",
        "src_ip",
        "base_domain",
    ]

    # Check for missing required columns
    missing_features = set(feature_cols) - set(df_feat.columns)
    if missing_features:
        raise ValueError(f"Failed to create features: {missing_features}")

    df_feat = df_feat[feature_cols]

    # Drop rows with null labels
    initial_rows = len(df_feat)
    df_feat = df_feat.dropna(subset=["label"])
    dropped_label = initial_rows - len(df_feat)

    if dropped_label > 0:
        logger.warning(f"Dropped {dropped_label} rows with missing labels")

    # Drop rows missing critical numeric features
    numeric_cols = [
        "qname_entropy",
        "qname_length",
        "numeric_ratio",
        "subdomain_depth",
        "iat_seconds",
    ]

    initial_rows = len(df_feat)
    df_feat = df_feat.dropna(subset=numeric_cols)
    dropped_numeric = initial_rows - len(df_feat)

    if dropped_numeric > 0:
        logger.warning(f"Dropped {dropped_numeric} rows with missing numeric features")

    # Data type coercion
    df_feat = df_feat.astype(
        {
            "qname_entropy": np.float32,
            "qname_length": np.int32,
            "numeric_ratio": np.float32,
            "subdomain_depth": np.int32,
            "qtype": np.int32,
            "label": np.int8,
            "iat_seconds": np.float32,
        }
    )

    logger.info(
        f"Feature extraction complete: {len(df_feat)} records, "
        f"{len([c for c in df_feat.columns if 'src_ip' not in c and 'base_domain' not in c])} features"
    )

    return df_feat


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    sample_data = pd.DataFrame(
        {
            "qname": [
                "example.com",
                "sub.example.com",
                "very.long.subdomain.example.com",
                "malware123.com",
            ],
            "qtype": [1, 1, 1, 28],
            "label": [0, 0, 1, 1],
            "timestamp": [100.0, 101.0, 105.0, 200.0],
            "src_ip": ["192.168.1.1", "192.168.1.1", "192.168.1.2", "192.168.1.2"],
        }
    )

    features_df = extract_dns_features(sample_data)
    print(features_df)
