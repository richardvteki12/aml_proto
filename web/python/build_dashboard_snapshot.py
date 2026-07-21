"""Build the Page 1 JSON snapshot from the AML project artifacts.

This script does not train or alter the anomaly model.  It only reads the
already-generated raw data, ABT, ground truth, and ML evaluation artifacts and
writes a lightweight JSON file that the Next.js evaluation page can render.

Run from ``web`` with:
    E:\\Anaconda3\\envs\\super\\python.exe python\\build_dashboard_snapshot.py
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ML_DIR = PROCESSED_DIR / "ml_anomaly_detection"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "ground_truth" / "aml_ground_truth.csv"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "aml_anomaly_detection" / "model_metadata.json"
OUTPUT_PATH = PROJECT_ROOT / "web" / "public" / "dashboard-data.json"

ACTIVE_SCENARIOS = ["AML-S01", "AML-S02", "AML-S03", "AML-S04", "AML-S05"]
RULES = [
    ("RB01", "Structuring / Smurfing", "AML-S01", "is_structuring_candidate_24h", "High"),
    ("RB02", "Sudden Transaction Spike", "AML-S02", "is_sudden_spike_candidate", "High"),
    ("RB03", "Rapid Movement of Funds", "AML-S03", "is_rapid_movement_candidate", "High"),
    ("RB04", "Dormant Account Reactivation", "AML-S04", "is_dormant_reactivation_candidate", "High"),
    ("RB05", "Multiple Senders to One Receiver", "AML-S05", "is_multiple_senders_candidate_24h", "Medium"),
]


def count_csv_rows(path: Path) -> int:
    """Count data rows without loading the full CSV into memory."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def number(value: object, digits: int = 6) -> float | int:
    """Convert numpy/pandas scalar to a compact JSON number."""

    as_float = float(value)
    if as_float.is_integer():
        return int(as_float)
    return round(as_float, digits)


def pct(numerator: int | float, denominator: int | float) -> float:
    return round((float(numerator) / float(denominator) * 100) if denominator else 0.0, 4)


def row_to_metrics(row: pd.Series) -> dict[str, float | int | str]:
    """Keep the metric names stable between baseline and tuned-model tables."""

    return {
        "modelName": str(row["model_name"]),
        "rocAuc": number(row["roc_auc"]),
        "averagePrecision": number(row["average_precision"]),
        "apLift": number(row["average_precision_lift_vs_random"]),
        "precisionAtTopK": number(row["precision_at_top_k"]),
        "recallAtTopK": number(row["recall_at_top_k"]),
        "topKRows": number(row["top_k_rows"]),
        "topKTruePositives": number(row["top_k_true_positives"]),
        "fitSeconds": number(row["fit_seconds"]),
    }


def main() -> None:
    abt_path = PROCESSED_DIR / "transaction_feature_abt.csv"
    flag_columns = [rule[3] for rule in RULES]
    # Only Page 1 aggregation columns are loaded from the 250k-row ABT.
    abt = pd.read_csv(abt_path, usecols=["transaction_id", "is_success", *flag_columns])
    abt[flag_columns] = abt[flag_columns].fillna(False).astype(bool)

    ground_truth = pd.read_csv(GROUND_TRUTH_PATH)
    active_truth = ground_truth.loc[ground_truth["scenario_id"].isin(ACTIVE_SCENARIOS)].copy()
    truth_lookup = ground_truth.set_index("transaction_id")

    rule_rows: list[dict[str, object]] = []
    for rule_id, rule_name, scenario_id, flag_column, severity in RULES:
        flagged = abt.loc[abt[flag_column], "transaction_id"]
        matched_truth = truth_lookup.reindex(flagged).dropna(subset=["scenario_id"])
        own_truth = ground_truth.loc[ground_truth["scenario_id"].eq(scenario_id), "transaction_id"]
        own_hits = int(flagged.isin(own_truth).sum())
        rule_rows.append(
            {
                "id": rule_id,
                "name": rule_name,
                "scenarioId": scenario_id,
                "severity": severity,
                "candidateHits": int(len(flagged)),
                "candidateRatePct": pct(len(flagged), len(abt)),
                "groundTruthTransactions": int(len(own_truth)),
                "ownTypologyTruePositiveHits": own_hits,
                "recallPct": pct(own_hits, len(own_truth)),
                "allGroundTruthHits": int(len(matched_truth)),
            }
        )

    any_rule = abt[flag_columns].any(axis=1)
    active_truth_ids = set(active_truth["transaction_id"])
    all_truth_ids = set(ground_truth["transaction_id"])
    any_rule_ids = set(abt.loc[any_rule, "transaction_id"])

    baseline = pd.read_csv(ML_DIR / "baseline_model_comparison.csv")
    tuning = pd.read_csv(ML_DIR / "winner_hyperparameter_tuning.csv")
    typology_recall = pd.read_csv(ML_DIR / "test_recall_by_typology.csv")
    test_scores = pd.read_csv(
        ML_DIR / "test_scored_transactions.csv",
        usecols=["known_aml_label", "any_rule_alert", "is_top_k_alert"],
    )
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    final_test = metadata["final_test_metrics"]
    selected = tuning.sort_values(["average_precision", "roc_auc"], ascending=False).iloc[0]

    test_known = test_scores.loc[test_scores["known_aml_label"].eq(1)]
    rule_missed = test_known.loc[test_known["any_rule_alert"].eq(0)]
    recovered = rule_missed.loc[rule_missed["is_top_k_alert"].eq(1)]
    combined = test_known.loc[test_known["any_rule_alert"].eq(1) | test_known["is_top_k_alert"].eq(1)]

    scenario_rows = []
    for scenario_id, group in ground_truth.groupby("scenario_id", sort=True):
        scenario_rows.append(
            {
                "scenarioId": scenario_id,
                "name": str(group["scenario_name"].iloc[0]),
                "transactions": int(len(group)),
                "customers": int(group["customer_id"].nunique()),
                "scenarioGroups": int(group["scenario_group_id"].nunique()),
                "inActiveScope": scenario_id in ACTIVE_SCENARIOS,
            }
        )

    output = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceNote": "Dihitung dari artefak CSV dan model metadata yang sudah ada; tidak ada training saat page dibuka.",
        "overview": {
            "customers": count_csv_rows(RAW_DIR / "customers.csv"),
            "accounts": count_csv_rows(RAW_DIR / "accounts.csv"),
            "counterparties": count_csv_rows(RAW_DIR / "counterparties.csv"),
            "transactions": int(len(abt)),
            "successfulTransactions": int(abt["is_success"].sum()),
            "abtColumns": len(pd.read_csv(abt_path, nrows=0).columns),
            "allGroundTruthTransactions": int(len(ground_truth)),
            "activeScopeGroundTruthTransactions": int(len(active_truth)),
        },
        "groundTruth": {
            "allInjectedTransactions": int(len(ground_truth)),
            "activeScopeTransactions": int(len(active_truth)),
            "activeScenarios": ACTIVE_SCENARIOS,
            "scenarios": scenario_rows,
        },
        "rules": {
            "population": int(len(abt)),
            "anyRuleCandidateHits": int(any_rule.sum()),
            "candidateRatePct": pct(any_rule.sum(), len(abt)),
            "activeScopeHits": int(len(any_rule_ids.intersection(active_truth_ids))),
            "activeScopeRecallPct": pct(len(any_rule_ids.intersection(active_truth_ids)), len(active_truth_ids)),
            "allGroundTruthHits": int(len(any_rule_ids.intersection(all_truth_ids))),
            "allGroundTruthRecallPct": pct(len(any_rule_ids.intersection(all_truth_ids)), len(all_truth_ids)),
            "items": rule_rows,
        },
        "ml": {
            "modelName": metadata["model_name"],
            "scope": metadata["population"],
            "reviewTopFraction": metadata["review_top_fraction"],
            "rawFeatureColumns": 40,
            "selectedParameters": metadata["selected_parameters"],
            "baseline": [row_to_metrics(row) for _, row in baseline.iterrows()],
            "tunedWinner": row_to_metrics(selected),
            "finalTest": {
                "rows": int(final_test["rows"]),
                "knownAmlPositives": int(final_test["known_aml_positives"]),
                "rocAuc": number(final_test["roc_auc"]),
                "averagePrecision": number(final_test["average_precision"]),
                "apLift": number(final_test["average_precision_lift_vs_random"]),
                "topKRows": int(final_test["top_k_rows"]),
                "topKTruePositives": int(final_test["top_k_true_positives"]),
                "precisionAtTopKPct": pct(final_test["top_k_true_positives"], final_test["top_k_rows"]),
                "recallAtTopKPct": pct(final_test["top_k_true_positives"], final_test["known_aml_positives"]),
                "ruleMissedKnownAml": int(final_test["rule_missed_known_aml"]),
                "ruleMissRecoveredInTopK": int(final_test["rule_miss_recovered_in_top_k"]),
                "ruleMissRecoveryPct": pct(
                    final_test["rule_miss_recovered_in_top_k"], final_test["rule_missed_known_aml"]
                ),
            },
            "byTypology": [
                {
                    "scenarioId": str(row["scenario_id"]),
                    "name": str(row["scenario_name"]),
                    "knownAmlRows": int(row["known_aml_rows"]),
                    "topKHits": int(row["top_k_hits"]),
                    "recallAtTopKPct": pct(row["top_k_hits"], row["known_aml_rows"]),
                    "meanAnomalyScore": number(row["mean_anomaly_score"]),
                }
                for _, row in typology_recall.iterrows()
            ],
        },
        "hybrid": {
            "holdoutKnownAml": int(len(test_known)),
            "ruleOnlyCaptured": int(test_known["any_rule_alert"].sum()),
            "mlTopOnePctCaptured": int(test_known["is_top_k_alert"].sum()),
            "ruleMissed": int(len(rule_missed)),
            "ruleMissRecoveredByMl": int(len(recovered)),
            "combinedCaptured": int(len(combined)),
            "combinedRecallPct": pct(len(combined), len(test_known)),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Dashboard snapshot written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
