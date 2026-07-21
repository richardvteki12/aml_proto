"""CLI bridge from the Next.js route to the already-trained AML model.

One JSON object is read from standard input and one JSON response is written to
standard output.  This module never calls ``fit``: it only loads
``best_anomaly_model.joblib`` and invokes its stored preprocessing + LOF scorer.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.aml_ml_features import (  # noqa: E402 - project root is added above.
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    LOG_NUMERIC_FEATURES,
    RAW_FEATURE_COLUMNS,
)


MODEL_PATH = PROJECT_ROOT / "models" / "aml_anomaly_detection" / "best_anomaly_model.joblib"
TEST_SCORES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "ml_anomaly_detection" / "test_scored_transactions.csv"
)
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "aml_anomaly_detection" / "model_metadata.json"

CATEGORY_OPTIONS = {
    "transaction_type": {"BI-FAST", "Cash", "RTGS", "SWIFT", "Transfer"},
    "channel": {"API", "ATM", "Branch", "Internet", "Mobile"},
    "currency": {"EUR", "IDR", "SGD", "USD"},
    "purpose_code": {"BILL", "FAMILY", "INVESTMENT", "OTHER", "SALARY", "TRADE"},
    "source_of_fund": {"Business", "Investment", "Salary", "Unknown"},
    "destination_country": {"AE", "AU", "GB", "ID", "JP", "KP", "MY", "RU", "SG", "SY", "US"},
    "sender_customer_segment": {"Corporate", "Priority", "Retail", "SME"},
    "sender_customer_risk_rating": {"High", "Low", "Medium"},
    "sender_account_type": {"Business", "Current", "Saving"},
    "sender_account_risk_level": {"High", "Low", "Medium"},
    "receiver_party_country": {"AE", "AU", "GB", "ID", "JP", "MY", "SG", "US"},
    "receiver_party_risk_level": {"High", "Low", "Medium"},
}


def as_number(record: dict[str, Any], name: str) -> float:
    value = record.get(name)
    if value is None or value == "":
        raise ValueError(f"Kolom {name} wajib diisi sebagai angka.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Kolom {name} harus berupa angka.") from error
    if not np.isfinite(numeric):
        raise ValueError(f"Kolom {name} harus berupa angka terbatas.")
    return numeric


def as_bool(record: dict[str, Any], name: str) -> int:
    value = record.get(name)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}:
        return 1
    if isinstance(value, str) and value.strip().lower() in {"false", "0", "no"}:
        return 0
    raise ValueError(f"Kolom {name} harus bernilai true/false.")


def normalise_record(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float | int]]:
    """Validate 40 raw features and derive two rule-only consistency values."""

    record: dict[str, Any] = {}
    for name in LOG_NUMERIC_FEATURES:
        record[name] = as_number(payload, name)
    for name in BINARY_FEATURES:
        record[name] = as_bool(payload, name)

    record["transaction_hour"] = as_number(payload, "transaction_hour")
    if not 0 <= record["transaction_hour"] <= 23:
        raise ValueError("transaction_hour harus berada pada rentang 0–23.")

    for name in CATEGORICAL_FEATURES:
        value = str(payload.get(name, "")).strip()
        if value not in CATEGORY_OPTIONS[name]:
            options = ", ".join(sorted(CATEGORY_OPTIONS[name]))
            raise ValueError(f"{name} harus salah satu dari: {options}.")
        record[name] = value

    if record["amount_idr_equivalent"] < 0:
        raise ValueError("amount_idr_equivalent tidak boleh negatif.")

    # These two helper fields are not ML inputs.  They make manual inference
    # coherent: a stated inbound amount determines the rapid-movement flag and
    # its outbound/inbound ratio; the stored pipeline receives the ratio.
    inbound_amount = as_number(payload, "last_internal_inbound_amount_idr")
    prior_amount_max = as_number(payload, "prior_amount_max_30d_idr")
    if inbound_amount < 0 or prior_amount_max < 0:
        raise ValueError("Nominal historis tidak boleh negatif.")

    if inbound_amount > 0:
        record["has_prior_internal_inbound_24h"] = 1
        record["outbound_to_last_inbound_ratio"] = record["amount_idr_equivalent"] / inbound_amount
    else:
        record["has_prior_internal_inbound_24h"] = 0
        record["outbound_to_last_inbound_ratio"] = 0.0

    helpers = {
        "is_success": as_bool(payload, "is_success"),
        "last_internal_inbound_amount_idr": inbound_amount,
        "prior_amount_max_30d_idr": prior_amount_max,
        "amount_to_prior_max_ratio_30d": (
            record["amount_idr_equivalent"] / prior_amount_max if prior_amount_max > 0 else 0.0
        ),
    }

    missing = sorted(set(RAW_FEATURE_COLUMNS).difference(record))
    if missing:
        raise ValueError("Feature kontrak tidak lengkap: " + ", ".join(missing))
    return record, helpers


def explain_rules(record: dict[str, Any], helpers: dict[str, float | int]) -> list[dict[str, Any]]:
    """Evaluate the five current AML red-flag candidates transparently."""

    success = bool(helpers["is_success"])
    amount = float(record["amount_idr_equivalent"])
    rules = [
        {
            "id": "RB01",
            "name": "Structuring / Smurfing",
            "severity": "High",
            "hit": success
            and amount <= 10_000_000
            and record["sender_subthreshold_txn_count_24h"] >= 4
            and record["sender_subthreshold_amount_sum_24h_idr"] >= 30_000_000,
            "reason": (
                f"Nominal transaksi Rp{amount:,.0f}; dalam 24 jam ada "
                f"{record['sender_subthreshold_txn_count_24h']:,.0f} transaksi sub-threshold "
                f"dengan total Rp{record['sender_subthreshold_amount_sum_24h_idr']:,.0f}."
            ),
        },
        {
            "id": "RB02",
            "name": "Rapid Movement of Funds",
            "severity": "High",
            "hit": success
            and record["has_prior_internal_inbound_24h"] == 1
            and 0 < record["minutes_since_last_internal_inbound"] <= 30
            and 0.75 <= record["outbound_to_last_inbound_ratio"] <= 1.10,
            "reason": (
                f"Dana masuk internal terakhir Rp{helpers['last_internal_inbound_amount_idr']:,.0f}; "
                f"jeda {record['minutes_since_last_internal_inbound']:,.1f} menit dan rasio keluar/masuk "
                f"{record['outbound_to_last_inbound_ratio']:.2f}."
            ),
        },
        {
            "id": "RB03",
            "name": "Sudden Transaction Spike",
            "severity": "High",
            "hit": success
            and record["has_sufficient_history_30d"] == 1
            and helpers["amount_to_prior_max_ratio_30d"] >= 3.0,
            "reason": (
                f"Nominal saat ini {helpers['amount_to_prior_max_ratio_30d']:.2f}× maksimum "
                f"nominal historis 30 hari (Rp{helpers['prior_amount_max_30d_idr']:,.0f})."
            ),
        },
        {
            "id": "RB04",
            "name": "Dormant Account Reactivation",
            "severity": "High",
            "hit": success
            and record["has_prior_successful_sender_activity"] == 1
            and record["days_since_prior_successful_sender_activity"] >= 60
            and amount >= 150_000_000,
            "reason": (
                f"Jeda sejak aktivitas berhasil sebelumnya "
                f"{record['days_since_prior_successful_sender_activity']:.1f} hari; "
                f"nominal transaksi Rp{amount:,.0f}."
            ),
        },
        {
            "id": "RB05",
            "name": "Multiple Senders to One Receiver",
            "severity": "Medium",
            "hit": success
            and record["receiver_txn_count_24h"] >= 4
            and record["distinct_senders_to_receiver_24h"] >= 4,
            "reason": (
                f"Penerima menerima {record['receiver_txn_count_24h']:,.0f} transaksi dari "
                f"{record['distinct_senders_to_receiver_24h']:,.0f} pengirim unik dalam 24 jam."
            ),
        },
    ]

    for rule in rules:
        if not success:
            rule["reason"] = "Status transaksi tidak berhasil; rule aktif dan model saat ini dievaluasi untuk transaksi berhasil."
        elif not rule["hit"]:
            rule["reason"] += " Threshold rule belum terpenuhi."
    return rules


def score_against_reference(record: dict[str, Any]) -> dict[str, Any]:
    """Score one row with the fitted artifact and calibrate only for display."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(MODEL_PATH)
    frame = pd.DataFrame([record])
    score = float(bundle.anomaly_score(frame)[0])
    reference = pd.read_csv(TEST_SCORES_PATH, usecols=["anomaly_score"])["anomaly_score"].to_numpy()
    percentile = float((reference <= score).mean() * 100)
    p50, p95, p99, p995, p999 = np.percentile(reference, [50, 95, 99, 99.5, 99.9])

    if score >= p99:
        band = "Sangat tidak lazim"
        explanation = "Skor berada pada atau di atas persentil 99 data holdout referensi."
    elif score >= p95:
        band = "Tidak lazim"
        explanation = "Skor berada pada atau di atas persentil 95 data holdout referensi."
    else:
        band = "Dalam kisaran umum"
        explanation = "Skor belum mencapai persentil 95 data holdout referensi."

    return {
        "modelName": str(bundle.model_name),
        "score": round(score, 6),
        "referencePercentile": round(percentile, 3),
        "band": band,
        "explanation": explanation,
        "reviewPolicy": "Dalam produksi, urutkan satu batch transaksi dan review top 1%; score tunggal bukan probabilitas AML atau threshold final.",
        "referenceDistribution": {
            "p50": round(float(p50), 6),
            "p95": round(float(p95), 6),
            "p99": round(float(p99), 6),
            "p995": round(float(p995), 6),
            "p999": round(float(p999), 6),
        },
    }


def run(payload: dict[str, Any]) -> dict[str, Any]:
    record, helpers = normalise_record(payload)
    rules = explain_rules(record, helpers)
    hits = [rule for rule in rules if rule["hit"]]

    if not helpers["is_success"]:
        return {
            "modelUsed": "best_anomaly_model.joblib",
            "modelScopeWarning": "Model LOF ini hanya dilatih untuk transaksi berhasil. Tidak ada ML score untuk status gagal/reversed.",
            "rules": rules,
            "ruleHitCount": len(hits),
            "ml": None,
        }

    return {
        "modelUsed": "best_anomaly_model.joblib",
        "modelScopeWarning": None,
        "rules": rules,
        "ruleHitCount": len(hits),
        "ml": score_against_reference(record),
    }


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        response = run(payload)
        print(json.dumps({"ok": True, "data": response}))
    except Exception as error:  # noqa: BLE001 - bridge must serialize errors for the route.
        print(json.dumps({"ok": False, "error": str(error)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
