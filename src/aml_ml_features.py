"""Reusable, inference-safe feature preparation for AML anomaly models.

The functions in this module deliberately accept the transaction-grain ABT,
not an already transformed NumPy matrix.  That design makes the saved model
usable later from Streamlit: the UI can pass an ABT-shaped dataframe to the
bundle and receive a consistent anomaly score.

Ground-truth columns and rule-output columns are intentionally absent from
this module.  They are evaluation artefacts, never ML input features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler


# ---------------------------------------------------------------------------
# 1. Feature contract
# ---------------------------------------------------------------------------
# These columns come from transaction_feature_abt.csv and are available at (or
# before) the transaction timestamp.  IDs, raw names, addresses, labels, and
# rule candidate flags are intentionally excluded.

LOG_NUMERIC_FEATURES = [
    "amount_idr_equivalent",
    "sender_customer_monthly_income",
    "sender_success_txn_count_1h",
    "sender_success_amount_sum_1h_idr",
    "sender_success_txn_count_120m",
    "sender_success_amount_sum_120m_idr",
    "sender_success_txn_count_24h",
    "sender_success_amount_sum_24h_idr",
    "sender_success_txn_count_7d",
    "sender_success_amount_sum_7d_idr",
    "sender_subthreshold_txn_count_24h",
    "sender_subthreshold_amount_sum_24h_idr",
    "minutes_since_last_internal_inbound",
    "outbound_to_last_inbound_ratio",
    "prior_success_txn_count_30d",
    "amount_to_prior_median_ratio_30d",
    "days_since_prior_successful_sender_activity",
    "receiver_txn_count_24h",
    "distinct_senders_to_receiver_24h",
    "receiver_amount_sum_24h_idr",
    "receiver_txn_count_7d",
    "distinct_senders_to_receiver_7d",
]

# A value of 0/1 is already semantically meaningful.  These fields do not
# need logarithmic transformation, but they still receive missing-value
# protection in the preprocessing pipeline.
BINARY_FEATURES = [
    "has_prior_internal_inbound_24h",
    "has_sufficient_history_30d",
    "has_prior_successful_sender_activity",
    "sender_customer_pep_flag",
    "is_internal_receiver",
]

# Raw hour is not used directly because hour 23 and hour 0 are neighbours, not
# opposite ends of a numeric line.  build_model_input creates the two cyclical
# representations below.
DERIVED_TIME_FEATURES = [
    "transaction_hour_sin",
    "transaction_hour_cos",
]

# These are deliberately low-cardinality business categories.  One-hot
# encoding lets a distance-based model such as LOF compare their effect without
# pretending that, for example, country code "US" is numerically larger than
# country code "ID".
CATEGORICAL_FEATURES = [
    "transaction_type",
    "channel",
    "currency",
    "purpose_code",
    "source_of_fund",
    "destination_country",
    "sender_customer_segment",
    "sender_customer_risk_rating",
    "sender_account_type",
    "sender_account_risk_level",
    "receiver_party_country",
    "receiver_party_risk_level",
]

RAW_FEATURE_COLUMNS = [
    *LOG_NUMERIC_FEATURES,
    *BINARY_FEATURES,
    "transaction_hour",
    *CATEGORICAL_FEATURES,
]

MODEL_INPUT_COLUMNS = [
    *LOG_NUMERIC_FEATURES,
    *BINARY_FEATURES,
    *DERIVED_TIME_FEATURES,
    *CATEGORICAL_FEATURES,
]

# In the ABT, -1 is a structural sentinel: it means that no prior event exists.
# It must not be treated as a literal negative duration by the model.
SENTINEL_TO_MISSING = [
    "minutes_since_last_internal_inbound",
    "days_since_prior_successful_sender_activity",
]


def _assert_required_columns(abt: pd.DataFrame) -> None:
    """Fail early with a readable message when the ABT contract changes."""

    missing = sorted(set(RAW_FEATURE_COLUMNS).difference(abt.columns))
    if missing:
        raise KeyError(
            "ABT tidak memiliki kolom yang dibutuhkan oleh model ML: "
            + ", ".join(missing)
        )


def build_model_input(abt: pd.DataFrame) -> pd.DataFrame:
    """Create the model-ready feature dataframe from an ABT-shaped dataframe.

    This function does *not* fit anything and does not read ground truth.  It
    performs only deterministic, transaction-time-safe work: column selection,
    conversion of structural sentinels to missing values, and cyclical hour
    features.  Statistical imputation/scaling is learned later on training data
    only by ``make_preprocessor``.
    """

    if not isinstance(abt, pd.DataFrame):
        raise TypeError("Input inference harus berupa pandas DataFrame.")

    _assert_required_columns(abt)

    model_input = abt.loc[:, RAW_FEATURE_COLUMNS].copy()

    # Convert numeric source columns safely.  Invalid future values become NaN
    # and are handled by the fitted training-data imputer, not silently coerced
    # to a business value such as zero.
    for column in [*LOG_NUMERIC_FEATURES, *BINARY_FEATURES, "transaction_hour"]:
        model_input[column] = pd.to_numeric(model_input[column], errors="coerce")

    # -1 means "no history", which differs from a genuine zero-minute/zero-day
    # gap.  Availability is preserved by the matching binary feature above.
    for column in SENTINEL_TO_MISSING:
        model_input.loc[model_input[column].eq(-1), column] = np.nan

    # Transaction hour is cyclical: 23:00 is close to 00:00.  Using sin/cos
    # prevents a false large distance between those two adjacent hours.
    hours = model_input["transaction_hour"].mod(24)
    model_input["transaction_hour_sin"] = np.sin(2 * np.pi * hours / 24)
    model_input["transaction_hour_cos"] = np.cos(2 * np.pi * hours / 24)
    model_input = model_input.drop(columns="transaction_hour")

    # Preserve genuine missing categorical values for SimpleImputer.  Object
    # dtype also prevents pandas from interpreting future category codes as a
    # numeric scale.
    for column in CATEGORICAL_FEATURES:
        model_input[column] = model_input[column].astype("object")

    return model_input.loc[:, MODEL_INPUT_COLUMNS]


def make_preprocessor() -> ColumnTransformer:
    """Return an unfitted preprocessing object for Isolation Forest and LOF.

    Both candidate models receive the exact same transformed matrix.  Log1p
    reduces the long right tails in amount/count/ratio features; RobustScaler is
    particularly important for LOF because its neighbourhood distances are
    sensitive to feature scale.  Isolation Forest can work without scaling, but
    sharing this robust pipeline makes the comparison fair and inference simple.
    """

    log_numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                # add_indicator keeps the information that an input was absent;
                # this is useful because absence of prior history can be a signal.
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            (
                "log1p",
                # All values in LOG_NUMERIC_FEATURES are non-negative after the
                # sentinel conversion.  log1p keeps zeros valid and compresses
                # extreme monetary/ratio values.
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            ),
            ("robust_scaler", RobustScaler()),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    time_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("robust_scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            (
                "one_hot",
                # Unknown future categories are ignored instead of crashing the
                # Streamlit inference flow.
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("log_numeric", log_numeric_pipeline, LOG_NUMERIC_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
            ("time", time_pipeline, DERIVED_TIME_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        # Dense output is intentional: both sklearn Isolation Forest and LOF
        # accept dense matrices, and the selected feature space is compact.
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def feature_schema() -> dict[str, list[str]]:
    """Return a JSON-serialisable description of the raw model contract."""

    return {
        "raw_feature_columns": list(RAW_FEATURE_COLUMNS),
        "model_input_columns": list(MODEL_INPUT_COLUMNS),
        "log_numeric_features": list(LOG_NUMERIC_FEATURES),
        "binary_features": list(BINARY_FEATURES),
        "derived_time_features": list(DERIVED_TIME_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "sentinel_to_missing": list(SENTINEL_TO_MISSING),
    }


@dataclass
class AMLAnomalyScoringBundle:
    """Serializable artifact used by the future Streamlit inference layer.

    ``anomaly_score`` accepts an ABT-shaped dataframe, applies the stored
    preprocessor, and returns scores where a *larger* number means more
    anomalous.  Both Isolation Forest and LOF expose lower raw scores for more
    unusual observations, so the sign is inverted in one central place.
    """

    model_name: str
    estimator: Any
    preprocessor: ColumnTransformer
    review_top_fraction: float
    feature_contract: dict[str, list[str]] = field(default_factory=feature_schema)
    reference_threshold: float | None = None
    training_note: str = ""

    def transform(self, abt: pd.DataFrame) -> np.ndarray:
        """Apply the stored, fitted preprocessing path to new ABT rows."""

        return np.asarray(self.preprocessor.transform(build_model_input(abt)))

    def anomaly_score(self, abt: pd.DataFrame) -> np.ndarray:
        """Return an anomaly score where higher means more suspicious."""

        transformed = self.transform(abt)
        raw_score = np.asarray(self.estimator.score_samples(transformed))
        return -raw_score

    def alert_mask(self, abt: pd.DataFrame) -> np.ndarray:
        """Apply the stored reference threshold when one is available."""

        if self.reference_threshold is None:
            raise ValueError(
                "Artifact ini belum memiliki reference_threshold. "
                "Gunakan ranking/top-fraction policy atau kalibrasi ulang threshold."
            )
        return self.anomaly_score(abt) >= self.reference_threshold
