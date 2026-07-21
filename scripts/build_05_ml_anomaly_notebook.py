"""Create the reader-facing AML anomaly-detection notebook.

This small builder is intentionally kept in ``scripts/`` so the generated
notebook can be regenerated without hand-editing Jupyter JSON.  Run it from the
project root after changing the notebook template below.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_ml_anomaly_detection.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


cells = [
    md(
        """
        # 05 — Unsupervised AML Anomaly Detection

        ## Goal

        Notebook ini membandingkan dua model unsupervised untuk mendeteksi anomali transaksi:

        1. **Isolation Forest** sebagai kandidat utama berbasis pohon acak.
        2. **Local Outlier Factor (LOF)** sebagai pembanding berbasis kepadatan lokal.

        Model dilatih **tanpa** ground-truth label. Ground truth hanya dipakai setelah scoring
        untuk memilih parameter pada validation set dan melaporkan hasil akhir pada untouched test set.

        **Input utama:** `data/processed/transaction_feature_abt.csv`<br>
        **Ground truth evaluasi:** `data/ground_truth/aml_ground_truth.csv`<br>
        **Output:** tabel perbandingan model, hasil scoring validation/test, dan artefak model `.joblib`
        untuk Streamlit inference nanti.
        """
    ),
    md(
        """
        ## Key assumptions and boundaries

        - Scope model saat ini hanya lima tipologi yang sudah memiliki feature engineering: AML-S01 sampai AML-S05.
        - Semua transaksi AML-S06 sampai AML-S10 dikeluarkan dari population ini. Mereka bukan negative label yang valid
          untuk scope lima tipologi dan beberapa scenario group-nya melintasi periode waktu.
        - Hanya transaksi `Success` yang discore. Lima tipologi saat ini menggambarkan perpindahan dana yang berhasil;
          failed/reversed retry behaviour dapat menjadi scope model terpisah di tahap berikutnya.
        - Split bersifat temporal: train → validation → test. Tidak ada `scenario_group_id` S01–S05 yang terpecah antar split.
        - `scenario_id`, `injected_flag`, candidate flag rule, dan seluruh ID tidak pernah masuk ke input ML.
        - Nilai non-ground-truth diperlakukan sebagai **synthetic baseline proxy**, bukan negative label operasional yang
          sudah dikonfirmasi oleh investigator.
        """
    ),
    md(
        """
        ## 1. Setup and imports

        Cell berikut mencari root project secara otomatis. Jangan mengubah current working directory secara manual;
        notebook dapat dijalankan dari folder `notebooks/` maupun project root.

        Modul `src/aml_ml_features.py` menyimpan kontrak feature dan transformer yang dipakai juga oleh model tersimpan.
        Menjaga modul ini bersama model adalah penting: `joblib.load()` di Streamlit perlu menemukan class
        `AMLAnomalyScoringBundle` yang sama.
        """
    ),
    code(
        """
        # Standard library: path handling, metadata, timing, and reproducibility.
        from __future__ import annotations

        import json
        import platform
        import sys
        from pathlib import Path
        from time import perf_counter

        # Data and numerical libraries.
        import joblib
        import numpy as np
        import pandas as pd

        # Notebook display helpers keep outputs small and readable.
        from IPython.display import Markdown, display

        # Unsupervised estimators and label-free ranking metrics for evaluation.
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.neighbors import LocalOutlierFactor


        def find_project_root() -> Path:
            '''Find the project root from either Jupyter's current folder or its parents.'''

            current = Path.cwd().resolve()
            for folder in [current, *current.parents]:
                if (
                    folder / "data" / "processed" / "transaction_feature_abt.csv"
                ).exists():
                    return folder

            raise FileNotFoundError(
                "Project root tidak ditemukan. Pastikan "
                "data/processed/transaction_feature_abt.csv sudah ada."
            )


        PROJECT_ROOT = find_project_root()

        # Allow the notebook to import the reusable inference module from src/.
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        from src.aml_ml_features import (
            AMLAnomalyScoringBundle,
            CATEGORICAL_FEATURES,
            DERIVED_TIME_FEATURES,
            LOG_NUMERIC_FEATURES,
            MODEL_INPUT_COLUMNS,
            RAW_FEATURE_COLUMNS,
            SENTINEL_TO_MISSING,
            BINARY_FEATURES,
            build_model_input,
            feature_schema,
            make_preprocessor,
        )


        print(f"Project root : {PROJECT_ROOT}")
        print(f"Python       : {sys.executable}")
        print(f"Platform     : {platform.platform()}")
        """
    ),
    md(
        """
        ## 2. Visible experiment configuration

        Semua keputusan yang memengaruhi hasil dibuat eksplisit di sini.

        `LOF_REFERENCE_SAMPLE_SIZE = None` berarti LOF memakai **seluruh train set** sebagai reference neighbourhood.
        Ini adalah konfigurasi paling lengkap, tetapi jauh lebih berat daripada Isolation Forest karena LOF perlu
        menghitung kedekatan transaksi terhadap banyak tetangga. Jalankan pada mesin yang memiliki RAM dan waktu cukup.

        `TOP_FRACTION = 0.01` berarti simulasi kapasitas review analyst adalah 1% transaksi dengan score tertinggi.
        Ini lebih bermakna daripada bergantung hanya pada parameter `contamination`.
        """
    ),
    code(
        """
        # Fixed random state makes Isolation Forest reproducible.
        RANDOM_STATE = 42

        # Only these five scenarios have explicit behavioural features in the current ABT.
        IN_SCOPE_SCENARIOS = [
            "AML-S01",  # Structuring / Smurfing
            "AML-S02",  # Sudden Transaction Spike
            "AML-S03",  # Rapid Movement of Funds
            "AML-S04",  # Dormant Account Reactivation
            "AML-S05",  # Multiple Senders to One Receiver
        ]

        # These cutoffs were checked against scenario_group_id. For S01-S05,
        # no synthetic event group crosses a split boundary.
        TRAIN_END = pd.Timestamp("2026-03-27 00:00:00")
        VALIDATION_END = pd.Timestamp("2026-05-14 00:00:00")

        # Operational review budget: select the highest anomaly scores only.
        TOP_FRACTION = 0.01

        # None means LOF uses the entire train set as its reference neighbourhood.
        # Set an integer only if a later engineering constraint requires sampling.
        LOF_REFERENCE_SAMPLE_SIZE = None
        LOF_ALGORITHM = "ball_tree"
        LOF_N_JOBS = 1  # Stable Windows setting; avoids thread-pool discovery errors.

        ABT_PATH = PROJECT_ROOT / "data" / "processed" / "transaction_feature_abt.csv"
        GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "ground_truth" / "aml_ground_truth.csv"
        RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "ml_anomaly_detection"
        MODEL_DIR = PROJECT_ROOT / "models" / "aml_anomaly_detection"

        display(
            pd.DataFrame(
                [
                    ("Train end (exclusive)", TRAIN_END),
                    ("Validation end (exclusive)", VALIDATION_END),
                    ("Review top fraction", TOP_FRACTION),
                    (
                        "LOF reference rows",
                        "FULL TRAIN SET"
                        if LOF_REFERENCE_SAMPLE_SIZE is None
                        else LOF_REFERENCE_SAMPLE_SIZE,
                    ),
                    ("LOF neighbour algorithm", LOF_ALGORITHM),
                    ("Random state", RANDOM_STATE),
                ],
                columns=["parameter", "value"],
            )
        )
        """
    ),
    md(
        """
        ## 3. Load ABT and ground truth separately

        Ground truth is loaded into a separate dataframe. It is not merged into the feature dataframe until the
        evaluation preparation cell below. This separation makes it clear that no label is available to either model's
        `.fit()` method.
        """
    ),
    code(
        """
        # Parse timestamp immediately because temporal splitting must use real datetime values.
        abt = pd.read_csv(ABT_PATH, parse_dates=["transaction_timestamp"])
        ground_truth = pd.read_csv(GROUND_TRUTH_PATH)

        # Transaction grain is a non-negotiable ABT contract.
        assert abt["transaction_id"].is_unique, "ABT harus tepat satu baris per transaction_id."
        assert ground_truth["transaction_id"].is_unique, "Ground truth tidak boleh menduplikasi transaction_id."
        assert abt["transaction_timestamp"].notna().all(), "Timestamp ABT tidak boleh kosong."

        # Every injected transaction must exist exactly once in ABT.
        gt_match = ground_truth[["transaction_id"]].merge(
            abt[["transaction_id"]],
            on="transaction_id",
            how="left",
            indicator=True,
            validate="one_to_one",
        )
        assert (gt_match["_merge"] == "both").all(), (
            "Ada transaction_id ground truth yang tidak ditemukan di ABT."
        )

        display(
            pd.DataFrame(
                [
                    ("ABT rows", len(abt)),
                    ("ABT columns", len(abt.columns)),
                    ("Unique transaction_id", abt["transaction_id"].nunique()),
                    ("Ground-truth rows", len(ground_truth)),
                    ("ABT date min", abt["transaction_timestamp"].min()),
                    ("ABT date max", abt["transaction_timestamp"].max()),
                ],
                columns=["check", "value"],
            )
        )
        """
    ),
    md(
        """
        ## 4. Define the ML evaluation population and temporal split

        The five in-scope scenarios become `known_aml_label = 1` **only for evaluation**. All other current
        in-scope population rows receive zero as a synthetic-baseline proxy.

        Out-of-scope injections AML-S06 to AML-S10 are removed before splitting. If they stayed in the data with label
        zero, the notebook would incorrectly teach evaluation that known suspicious transactions are normal.
        """
    ),
    code(
        """
        # Retain the ground-truth fields only as evaluation metadata.
        in_scope_ground_truth = ground_truth.loc[
            ground_truth["scenario_id"].isin(IN_SCOPE_SCENARIOS),
            ["transaction_id", "scenario_id", "scenario_name", "scenario_group_id"],
        ].copy()

        out_of_scope_ids = set(
            ground_truth.loc[
                ~ground_truth["scenario_id"].isin(IN_SCOPE_SCENARIOS),
                "transaction_id",
            ]
        )

        # Scope only successful transactions.  This avoids mixing failed/reversed
        # operational behaviour with completed-fund-flow AML typologies.
        model_data = abt.loc[
            abt["is_success"].astype(bool)
            & ~abt["transaction_id"].isin(out_of_scope_ids)
        ].copy()

        # The LEFT JOIN is evaluation-only. Rows without a scenario are baseline proxy rows.
        model_data = model_data.merge(
            in_scope_ground_truth,
            on="transaction_id",
            how="left",
            validate="one_to_one",
        )
        model_data["known_aml_label"] = model_data["scenario_id"].notna().astype(int)

        # Assign the split using information available before model fitting.
        model_data["data_split"] = np.select(
            [
                model_data["transaction_timestamp"] < TRAIN_END,
                model_data["transaction_timestamp"] < VALIDATION_END,
            ],
            ["train", "validation"],
            default="test",
        )

        # Sort once so any deterministic LOF reference sample covers the whole train timeline.
        model_data = model_data.sort_values("transaction_timestamp").reset_index(drop=True)

        # Candidate flags stay outside X.  They are used only later to measure whether
        # ML recovers known AML that the existing five rules did not alert.
        RULE_FLAG_COLUMNS = [
            "is_structuring_candidate_24h",
            "is_rapid_movement_candidate",
            "is_sudden_spike_candidate",
            "is_dormant_reactivation_candidate",
            "is_multiple_senders_candidate_24h",
        ]
        missing_flags = sorted(set(RULE_FLAG_COLUMNS).difference(model_data.columns))
        assert not missing_flags, f"Rule flag ABT tidak ditemukan: {missing_flags}"
        model_data["any_rule_alert"] = (
            model_data[RULE_FLAG_COLUMNS].max(axis=1).astype(int)
        )

        # Safety check: an injected multi-transaction pattern must not be split
        # between train/validation/test. This prevents scenario fragments leaking.
        group_split_check = (
            model_data.loc[model_data["known_aml_label"].eq(1)]
            .groupby("scenario_group_id")["data_split"]
            .nunique()
        )
        assert (group_split_check <= 1).all(), (
            "Leakage: satu scenario_group_id muncul pada lebih dari satu split."
        )

        split_summary = (
            model_data.groupby("data_split", sort=False)
            .agg(
                transactions=("transaction_id", "size"),
                known_aml_positives=("known_aml_label", "sum"),
                baseline_proxy_rows=("known_aml_label", lambda s: int((s == 0).sum())),
                positive_rate_pct=("known_aml_label", lambda s: s.mean() * 100),
                first_timestamp=("transaction_timestamp", "min"),
                last_timestamp=("transaction_timestamp", "max"),
            )
            .reindex(["train", "validation", "test"])
            .reset_index()
        )

        display(split_summary)
        print(f"Out-of-scope AML-S06–S10 excluded: {len(out_of_scope_ids):,} transactions")
        print(f"In-scope groups crossing splits: {(group_split_check > 1).sum():,}")
        """
    ),
    code(
        """
        # Display the positive rows per typology and split.  This table tells us
        # whether each evaluation period contains enough known injected patterns.
        scenario_split_summary = (
            model_data.loc[model_data["known_aml_label"].eq(1)]
            .pivot_table(
                index=["scenario_id", "scenario_name"],
                columns="data_split",
                values="transaction_id",
                aggfunc="size",
                fill_value=0,
            )
            .reindex(columns=["train", "validation", "test"], fill_value=0)
            .reset_index()
        )
        display(scenario_split_summary)
        """
    ),
    md(
        """
        ## 5. Feature contract: what enters the model and what stays out

        The feature builder selects behavioural, profile, and low-cardinality contextual fields only.

        It excludes IDs, names, addresses, raw timestamps, labels, rule candidate flags, and `debit_credit`.
        This prevents identity memorisation and makes the comparison between rule-based monitoring and ML fair.

        The `120m` features are retained as part of the initial feature set; they are not silently deleted. Later,
        a feature ablation can test whether they improve validation ranking beyond 1h/24h/7d.
        """
    ),
    code(
        """
        feature_catalogue = pd.DataFrame(
            [
                *[("Log + robust scaled numeric", column) for column in LOG_NUMERIC_FEATURES],
                *[("Binary context", column) for column in BINARY_FEATURES],
                *[("Derived cyclical time", column) for column in DERIVED_TIME_FEATURES],
                *[("One-hot categorical", column) for column in CATEGORICAL_FEATURES],
            ],
            columns=["feature_family", "feature"],
        )

        display(feature_catalogue)
        print(f"Raw ABT columns needed by feature builder : {len(RAW_FEATURE_COLUMNS)}")
        print(f"Model-input columns after time derivation  : {len(MODEL_INPUT_COLUMNS)}")
        print("Sentinel values converted to missing       :", SENTINEL_TO_MISSING)
        """
    ),
    code(
        """
        # build_model_input performs deterministic work only: it selects the
        # approved columns, converts -1 structural sentinels to NaN, and creates
        # hour_sin/hour_cos. It does NOT inspect labels or fit an imputer/scaler.
        raw_model_input = build_model_input(model_data)

        assert set(RULE_FLAG_COLUMNS).isdisjoint(raw_model_input.columns), (
            "Rule outputs must not enter ML features."
        )
        assert "known_aml_label" not in raw_model_input.columns, (
            "Ground truth label must not enter ML features."
        )
        assert raw_model_input.index.equals(model_data.index)

        # Numeric missing values at this point are allowed only as intentionally
        # converted structural sentinels. They will be learned/imputed from train.
        numeric_missing = raw_model_input.select_dtypes(include="number").isna().sum()
        display(
            numeric_missing[numeric_missing.gt(0)]
            .rename("missing_values_before_train_imputation")
            .to_frame()
        )
        display(raw_model_input.head(3))
        """
    ),
    md(
        """
        ## 6. Fit preprocessing using train data only

        `SimpleImputer`, `RobustScaler`, and `OneHotEncoder` are fitted exclusively on the train period.
        Validation and test rows only call `.transform()`. This matters because fitting a scaler on future values would
        leak information about their distribution into the past.

        Amounts, counts, sums, ratios, and durations use `log1p` plus `RobustScaler`. LOF is distance-based and
        therefore especially sensitive to unscaled money amounts. Isolation Forest does not strictly require scaling,
        but using the same pipeline makes the comparison consistent.
        """
    ),
    code(
        """
        # Keep the original ABT-shaped split data for model bundles/inference.
        train_data = model_data.loc[model_data["data_split"].eq("train")].copy()
        validation_data = model_data.loc[model_data["data_split"].eq("validation")].copy()
        test_data = model_data.loc[model_data["data_split"].eq("test")].copy()

        # Construct raw ML feature frames first, then fit preprocessing only on train.
        X_train_raw = build_model_input(train_data)
        X_validation_raw = build_model_input(validation_data)
        X_test_raw = build_model_input(test_data)

        fitted_preprocessor = make_preprocessor()
        X_train = fitted_preprocessor.fit_transform(X_train_raw)
        X_validation = fitted_preprocessor.transform(X_validation_raw)
        X_test = fitted_preprocessor.transform(X_test_raw)

        # A valid numerical matrix is required by both estimators.
        assert np.isfinite(X_train).all(), "Train matrix masih memiliki NaN/inf."
        assert np.isfinite(X_validation).all(), "Validation matrix masih memiliki NaN/inf."
        assert np.isfinite(X_test).all(), "Test matrix masih memiliki NaN/inf."

        transformed_feature_names = fitted_preprocessor.get_feature_names_out()

        display(
            pd.DataFrame(
                [
                    ("Train matrix", X_train.shape),
                    ("Validation matrix", X_validation.shape),
                    ("Test matrix", X_test.shape),
                    ("Transformed feature count", len(transformed_feature_names)),
                ],
                columns=["matrix", "shape_or_count"],
            )
        )
        display(pd.DataFrame({"transformed_feature": transformed_feature_names}).head(15))
        """
    ),
    md(
        """
        ## 7. Train the two baseline models

        - **Isolation Forest** is fitted on every train transaction.
        - **LOF** uses `novelty=True`, so it can score validation/test transactions that were not used as its reference set.
          Its reference sample is deterministic and time-spread, not label-selected.

        Neither `.fit()` receives `known_aml_label`, `scenario_id`, rule flags, or any ground truth field.
        """
    ),
    code(
        """
        # Baseline hyperparameters are intentionally modest. The purpose here is
        # an honest first comparison, not a large expensive search.
        BASELINE_PARAMS = {
            "Isolation Forest": {
                "n_estimators": 300,
                "max_samples": 256,
                "max_features": 1.0,
                "bootstrap": False,
                "contamination": "auto",
            },
            "Local Outlier Factor": {
                "n_neighbors": 35,
                "algorithm": LOF_ALGORITHM,
                "leaf_size": 40,
                "contamination": "auto",
            },
        }


        def temporal_reference_indices(
            total_rows: int, maximum_rows: int | None
        ) -> np.ndarray:
            '''Return all train rows or a deterministic time-spread reference sample.

            X_train was sorted by transaction timestamp before feature creation.
            Evenly spaced positions therefore cover early, middle, and late train
            history without using a label to decide which rows are retained.
            '''

            # None is the explicit full-train-set mode requested for LOF.
            if maximum_rows is None:
                return np.arange(total_rows)

            if total_rows <= maximum_rows:
                return np.arange(total_rows)
            return np.linspace(0, total_rows - 1, maximum_rows, dtype=int)


        def make_estimator(model_name: str, params: dict):
            '''Construct one estimator without fitting it.'''

            if model_name == "Isolation Forest":
                return IsolationForest(
                    **params,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )

            if model_name == "Local Outlier Factor":
                # novelty=True is essential: LOF must score unseen validation/test
                # transactions, not only the rows it fitted on.
                return LocalOutlierFactor(
                    **params,
                    novelty=True,
                    n_jobs=LOF_N_JOBS,
                )

            raise ValueError(f"Model tidak dikenal: {model_name}")


        def fit_estimator(model_name: str, params: dict, train_matrix: np.ndarray):
            '''Fit one model and return estimator, fit rows, and elapsed seconds.'''

            estimator = make_estimator(model_name, params)
            fit_matrix = train_matrix

            if model_name == "Local Outlier Factor":
                # When LOF_REFERENCE_SAMPLE_SIZE is None, reference_index contains
                # every train row. There is no hidden sampling in full-train mode.
                reference_index = temporal_reference_indices(
                    len(train_matrix), LOF_REFERENCE_SAMPLE_SIZE
                )
                fit_matrix = train_matrix[reference_index]

            started = perf_counter()
            estimator.fit(fit_matrix)
            elapsed_seconds = perf_counter() - started
            return estimator, len(fit_matrix), elapsed_seconds


        def anomaly_score(estimator, transformed_matrix: np.ndarray) -> np.ndarray:
            '''Return a consistently oriented score: larger = more anomalous.

            Both sklearn estimators emit lower raw score for a more unusual row,
            so one minus-sign is applied centrally and documented here.
            '''

            return -np.asarray(estimator.score_samples(transformed_matrix), dtype=float)


        baseline_estimators = {}
        baseline_rows = []
        validation_scored = validation_data[
            [
                "transaction_id",
                "transaction_timestamp",
                "scenario_id",
                "scenario_name",
                "scenario_group_id",
                "known_aml_label",
                "any_rule_alert",
            ]
        ].copy()

        for model_name, params in BASELINE_PARAMS.items():
            estimator, fit_rows, elapsed_seconds = fit_estimator(
                model_name, params, X_train
            )
            scores = anomaly_score(estimator, X_validation)

            baseline_estimators[model_name] = estimator
            validation_scored[model_name] = scores
            baseline_rows.append(
                {
                    "model_name": model_name,
                    "fit_rows": fit_rows,
                    "fit_seconds": elapsed_seconds,
                    "minimum_score": float(scores.min()),
                    "maximum_score": float(scores.max()),
                }
            )

        baseline_fit_summary = pd.DataFrame(baseline_rows)
        display(baseline_fit_summary)
        """
    ),
    md(
        """
        ## 8. Evaluate anomaly ranking on validation data

        Anomaly detection should be assessed as a **ranking** problem. The model gives every transaction a score;
        analyst capacity determines how many top rows are reviewed.

        Main metrics:

        - `average_precision`: prioritises rare known AML better than raw accuracy.
        - `recall_at_top_k`: known AML captured within the analyst review budget.
        - `precision_at_top_k`: quality of the reviewed queue under synthetic labels.
        - `rule_miss_recovery`: known AML missed by all five active rules but recovered by ML in Top-K.

        ROC-AUC is included as a secondary ranking metric, not the sole business decision.
        """
    ),
    code(
        """
        def ranking_metrics(
            y_true: pd.Series | np.ndarray,
            scores: pd.Series | np.ndarray,
            top_fraction: float,
            rule_alerts: pd.Series | np.ndarray | None = None,
        ) -> tuple[dict, np.ndarray]:
            '''Calculate label-free-model ranking metrics after scoring.

            `y_true` arrives only in this evaluation function. It is never passed
            to Isolation Forest or LOF during fitting.
            '''

            y = np.asarray(y_true, dtype=int)
            score_array = np.asarray(scores, dtype=float)
            assert len(y) == len(score_array)
            assert np.isfinite(score_array).all()

            total_rows = len(y)
            total_positives = int(y.sum())
            if total_positives == 0:
                raise ValueError("Metrik tidak dapat dihitung tanpa positive ground truth.")

            # Stable sort makes ties deterministic and therefore reproducible.
            order = np.argsort(-score_array, kind="mergesort")
            top_k = max(1, int(np.ceil(total_rows * top_fraction)))
            top_mask = np.zeros(total_rows, dtype=bool)
            top_mask[order[:top_k]] = True

            top_true_positives = int(y[top_mask].sum())
            metrics = {
                "rows": total_rows,
                "known_aml_positives": total_positives,
                "positive_rate_pct": float(y.mean() * 100),
                "roc_auc": float(roc_auc_score(y, score_array)),
                "average_precision": float(average_precision_score(y, score_array)),
                "random_precision_baseline": float(y.mean()),
                "top_fraction": top_fraction,
                "top_k_rows": top_k,
                "top_k_true_positives": top_true_positives,
                "precision_at_top_k": float(top_true_positives / top_k),
                "recall_at_top_k": float(top_true_positives / total_positives),
            }
            metrics["average_precision_lift_vs_random"] = (
                metrics["average_precision"] / metrics["random_precision_baseline"]
            )

            if rule_alerts is not None:
                rule_array = np.asarray(rule_alerts, dtype=int)
                rule_missed_positive = (y == 1) & (rule_array == 0)
                rule_missed_count = int(rule_missed_positive.sum())
                recovered_count = int((rule_missed_positive & top_mask).sum())

                metrics["rule_missed_known_aml"] = rule_missed_count
                metrics["rule_miss_recovered_in_top_k"] = recovered_count
                metrics["rule_miss_recovery_at_top_k"] = (
                    float(recovered_count / rule_missed_count)
                    if rule_missed_count
                    else np.nan
                )

            return metrics, top_mask


        validation_metric_rows = []
        validation_top_masks = {}

        for model_name in BASELINE_PARAMS:
            metrics, top_mask = ranking_metrics(
                validation_scored["known_aml_label"],
                validation_scored[model_name],
                TOP_FRACTION,
                validation_scored["any_rule_alert"],
            )
            metrics["model_name"] = model_name
            metrics["fit_seconds"] = float(
                baseline_fit_summary.loc[
                    baseline_fit_summary["model_name"].eq(model_name), "fit_seconds"
                ].iloc[0]
            )
            validation_metric_rows.append(metrics)
            validation_top_masks[model_name] = top_mask

        baseline_comparison = (
            pd.DataFrame(validation_metric_rows)
            .sort_values(
                ["average_precision", "recall_at_top_k", "roc_auc"],
                ascending=[False, False, False],
            )
            .reset_index(drop=True)
        )

        display(
            baseline_comparison[
                [
                    "model_name",
                    "roc_auc",
                    "average_precision",
                    "average_precision_lift_vs_random",
                    "recall_at_top_k",
                    "precision_at_top_k",
                    "rule_miss_recovery_at_top_k",
                    "top_k_rows",
                    "fit_seconds",
                ]
            ]
        )

        # The validation winner alone proceeds to the compact hyperparameter search.
        baseline_winner = baseline_comparison.iloc[0]["model_name"]
        print(f"Validation baseline winner: {baseline_winner}")
        """
    ),
    code(
        """
        def scenario_recall_at_top_k(scored_frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
            '''Show global Top-K recall separately for each in-scope typology.'''

            _, top_mask = ranking_metrics(
                scored_frame["known_aml_label"],
                scored_frame[score_column],
                TOP_FRACTION,
            )
            rows = []

            for scenario_id in IN_SCOPE_SCENARIOS:
                scenario_rows = scored_frame.loc[
                    scored_frame["scenario_id"].eq(scenario_id)
                ]
                top_hits = int(top_mask[scenario_rows.index.to_numpy() - scored_frame.index.min()].sum())

                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_name": scenario_rows["scenario_name"].iloc[0]
                        if len(scenario_rows)
                        else "not present in this split",
                        "known_aml_rows": len(scenario_rows),
                        "top_k_hits": top_hits,
                        "recall_at_top_k": top_hits / len(scenario_rows)
                        if len(scenario_rows)
                        else np.nan,
                        "mean_anomaly_score": scenario_rows[score_column].mean()
                        if len(scenario_rows)
                        else np.nan,
                    }
                )

            return pd.DataFrame(rows)


        # Reset index before calling the helper so the Top-K mask aligns exactly.
        validation_scored = validation_scored.reset_index(drop=True)

        for model_name in BASELINE_PARAMS:
            print(f"\\n{model_name} — recall per typology on validation")
            display(scenario_recall_at_top_k(validation_scored, model_name))
        """
    ),
    md(
        """
        ## 9. Tune the validation winner only

        This is not ordinary supervised `GridSearchCV` with accuracy. Each candidate is fitted without labels,
        scored on validation transactions, then compared using the same ranking metrics.

        `contamination` remains `"auto"` because the operational alert rate is controlled transparently through
        `TOP_FRACTION`, rather than being hidden inside the model's binary prediction threshold.
        """
    ),
    code(
        """
        # Compact, explainable search spaces. Expand later only if the initial
        # winner is promising and runtime permits it.
        TUNING_CANDIDATES = {
            "Isolation Forest": [
                {"n_estimators": 300, "max_samples": 256, "max_features": 1.0, "bootstrap": False, "contamination": "auto"},
                {"n_estimators": 500, "max_samples": 256, "max_features": 1.0, "bootstrap": False, "contamination": "auto"},
                {"n_estimators": 500, "max_samples": 512, "max_features": 1.0, "bootstrap": False, "contamination": "auto"},
                {"n_estimators": 500, "max_samples": 512, "max_features": 0.7, "bootstrap": False, "contamination": "auto"},
            ],
            "Local Outlier Factor": [
                {"n_neighbors": 20, "algorithm": LOF_ALGORITHM, "leaf_size": 40, "contamination": "auto"},
                {"n_neighbors": 35, "algorithm": LOF_ALGORITHM, "leaf_size": 40, "contamination": "auto"},
                {"n_neighbors": 50, "algorithm": LOF_ALGORITHM, "leaf_size": 40, "contamination": "auto"},
            ],
        }


        def parameter_key(params: dict) -> str:
            '''Create one stable text key for fitted-estimator lookup and CSV output.'''

            return json.dumps(params, sort_keys=True)


        tuning_rows = []
        tuned_estimators = {}

        for params in TUNING_CANDIDATES[baseline_winner]:
            estimator, fit_rows, elapsed_seconds = fit_estimator(
                baseline_winner, params, X_train
            )
            scores = anomaly_score(estimator, X_validation)
            metrics, _ = ranking_metrics(
                validation_scored["known_aml_label"],
                scores,
                TOP_FRACTION,
                validation_scored["any_rule_alert"],
            )

            key = parameter_key(params)
            tuned_estimators[key] = estimator
            tuning_rows.append(
                {
                    "model_name": baseline_winner,
                    "parameter_key": key,
                    "parameters": params,
                    "fit_rows": fit_rows,
                    "fit_seconds": elapsed_seconds,
                    **metrics,
                }
            )

        tuning_results = (
            pd.DataFrame(tuning_rows)
            .sort_values(
                ["average_precision", "recall_at_top_k", "roc_auc"],
                ascending=[False, False, False],
            )
            .reset_index(drop=True)
        )

        display(
            tuning_results[
                [
                    "model_name",
                    "parameters",
                    "average_precision",
                    "recall_at_top_k",
                    "precision_at_top_k",
                    "rule_miss_recovery_at_top_k",
                    "roc_auc",
                    "fit_seconds",
                ]
            ]
        )

        best_validation_row = tuning_results.iloc[0]
        best_params = best_validation_row["parameters"]
        best_validation_estimator = tuned_estimators[best_validation_row["parameter_key"]]

        print(f"Selected model      : {baseline_winner}")
        print(f"Selected parameters : {best_params}")
        """
    ),
    md(
        """
        ## 10. Final evaluation on the untouched test period

        The selected estimator below was fitted on train only and selected on validation only. Test data has not been
        used to choose a model, a feature set, parameter, or threshold. This is the reportable final experiment result.
        """
    ),
    code(
        """
        # Score the untouched test matrix with the already selected train-only estimator.
        test_scores = anomaly_score(best_validation_estimator, X_test)

        test_scored = test_data[
            [
                "transaction_id",
                "transaction_timestamp",
                "scenario_id",
                "scenario_name",
                "scenario_group_id",
                "known_aml_label",
                "any_rule_alert",
            ]
        ].copy().reset_index(drop=True)
        test_scored["anomaly_score"] = test_scores

        final_test_metrics, test_top_mask = ranking_metrics(
            test_scored["known_aml_label"],
            test_scored["anomaly_score"],
            TOP_FRACTION,
            test_scored["any_rule_alert"],
        )
        test_scored["is_top_k_alert"] = test_top_mask.astype(int)
        test_scored["anomaly_rank"] = (
            test_scored["anomaly_score"].rank(method="first", ascending=False).astype(int)
        )

        display(
            pd.DataFrame([final_test_metrics]).T.rename(columns={0: "value"})
        )
        """
    ),
    code(
        """
        # Per-typology recall answers a more useful question than one global
        # metric: which of the five intended AML behaviours is ranked highly?
        test_scenario_metrics = scenario_recall_at_top_k(
            test_scored, "anomaly_score"
        )
        display(test_scenario_metrics)

        # Show a small evidence table instead of a chart. Analysts can inspect
        # whether high-ranked transactions have understandable behavioural values.
        evidence_columns = [
            "transaction_id",
            "transaction_timestamp",
            "scenario_id",
            "scenario_name",
            "known_aml_label",
            "any_rule_alert",
            "anomaly_rank",
            "anomaly_score",
        ]
        display(
            test_scored.sort_values("anomaly_rank")
            .loc[:, evidence_columns]
            .head(15)
        )
        """
    ),
    md(
        """
        ## 11. Save models, feature contract, and scored outputs

        Two model bundles are saved deliberately:

        1. `validation_locked_model.joblib` — train-only model used for the reportable untouched-test result.
        2. `best_anomaly_model.joblib` — production model refitted on train + validation data, ready for Streamlit ranking.

        The production artifact uses a **Top 1% ranking policy**, not a fixed score threshold, because anomaly-score
        scales can shift after refitting. Streamlit can call `bundle.anomaly_score(abt_dataframe)`, rank the batch, then
        send the highest-scored fraction to review.
        """
    ),
    code(
        """
        # Create output folders only when this final persistence section runs.
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # A validation score threshold is valid only for the train-only model used
        # above. It is saved for reproducibility, not copied blindly to refitted production.
        selected_validation_scores = anomaly_score(best_validation_estimator, X_validation)
        validation_reference_threshold = float(
            np.quantile(selected_validation_scores, 1 - TOP_FRACTION)
        )

        validation_locked_bundle = AMLAnomalyScoringBundle(
            model_name=baseline_winner,
            estimator=best_validation_estimator,
            preprocessor=fitted_preprocessor,
            review_top_fraction=TOP_FRACTION,
            reference_threshold=validation_reference_threshold,
            training_note=(
                "Fitted on train period only; used for the locked untouched-test evaluation."
            ),
        )

        validation_locked_path = MODEL_DIR / "validation_locked_model.joblib"
        joblib.dump(validation_locked_bundle, validation_locked_path)

        # Refit the selected configuration with more unlabeled historical data for
        # the future Streamlit model. Ground truth is still not supplied to .fit().
        production_training_data = pd.concat(
            [train_data, validation_data], ignore_index=True
        ).sort_values("transaction_timestamp").reset_index(drop=True)

        production_preprocessor = make_preprocessor()
        X_production_train = production_preprocessor.fit_transform(
            build_model_input(production_training_data)
        )
        production_estimator, production_fit_rows, production_fit_seconds = fit_estimator(
            baseline_winner, best_params, X_production_train
        )

        production_bundle = AMLAnomalyScoringBundle(
            model_name=baseline_winner,
            estimator=production_estimator,
            preprocessor=production_preprocessor,
            review_top_fraction=TOP_FRACTION,
            # No absolute threshold after refit: Streamlit should rank the new
            # batch and use review_top_fraction as its operating policy.
            reference_threshold=None,
            training_note=(
                "Refitted on train plus validation without labels; use batch ranking/top fraction for inference."
            ),
        )

        production_model_path = MODEL_DIR / "best_anomaly_model.joblib"
        joblib.dump(production_bundle, production_model_path)

        # Save score outputs and documentation artifacts for presentation/audit.
        validation_scored.to_csv(RESULTS_DIR / "validation_scored_transactions.csv", index=False)
        test_scored.to_csv(RESULTS_DIR / "test_scored_transactions.csv", index=False)
        baseline_comparison.to_csv(RESULTS_DIR / "baseline_model_comparison.csv", index=False)
        tuning_results.to_csv(RESULTS_DIR / "winner_hyperparameter_tuning.csv", index=False)
        test_scenario_metrics.to_csv(RESULTS_DIR / "test_recall_by_typology.csv", index=False)

        with (MODEL_DIR / "model_feature_schema.json").open("w", encoding="utf-8") as handle:
            json.dump(feature_schema(), handle, indent=2)

        metadata = {
            "model_name": baseline_winner,
            "selected_parameters": best_params,
            "scope_scenarios": IN_SCOPE_SCENARIOS,
            "population": "successful transactions; AML-S06 to AML-S10 excluded",
            "temporal_split": {
                "train_end_exclusive": str(TRAIN_END),
                "validation_end_exclusive": str(VALIDATION_END),
            },
            "review_top_fraction": TOP_FRACTION,
            "lof_reference_sample_size": LOF_REFERENCE_SAMPLE_SIZE,
            "validation_reference_threshold_train_only": validation_reference_threshold,
            "final_test_metrics": final_test_metrics,
            "production_refit": {
                "rows_used": int(len(production_training_data)),
                "estimator_fit_rows": int(production_fit_rows),
                "fit_seconds": float(production_fit_seconds),
                "threshold_policy": "rank each incoming batch and select review_top_fraction",
            },
            "artifact_files": {
                "validation_locked_model": str(validation_locked_path.relative_to(PROJECT_ROOT)),
                "production_model": str(production_model_path.relative_to(PROJECT_ROOT)),
            },
        }
        with (MODEL_DIR / "model_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, default=str)

        artifact_files = pd.DataFrame(
            [
                {
                    "artifact": path.relative_to(PROJECT_ROOT).as_posix(),
                    "size_kb": round(path.stat().st_size / 1024, 1),
                }
                for path in sorted([*RESULTS_DIR.glob("*"), *MODEL_DIR.glob("*")])
                if path.is_file()
            ]
        )
        display(artifact_files)
        """
    ),
    md(
        """
        ## 12. Inference smoke test

        This test reloads the production artifact from disk and scores five previously unseen test transactions.
        It proves that the saved bundle contains both the fitted preprocessor and selected estimator.

        Keep `src/aml_ml_features.py` in the project when deploying Streamlit. The saved bundle imports its reusable
        `AMLAnomalyScoringBundle` class during `joblib.load()`.
        """
    ),
    code(
        """
        # Reload from disk instead of reusing the in-memory object. This is the
        # closest notebook-level simulation of how Streamlit will load the model.
        loaded_production_bundle = joblib.load(production_model_path)

        inference_sample = test_data.head(5).copy()
        inference_scores = loaded_production_bundle.anomaly_score(inference_sample)

        assert len(inference_scores) == len(inference_sample)
        assert np.isfinite(inference_scores).all(), "Inference menghasilkan NaN/inf."

        inference_preview = inference_sample[
            ["transaction_id", "transaction_timestamp", "amount_idr_equivalent"]
        ].copy()
        inference_preview["anomaly_score"] = inference_scores
        inference_preview["batch_rank"] = (
            inference_preview["anomaly_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )

        display(inference_preview.sort_values("batch_rank"))
        print("Inference smoke test passed: production model berhasil dimuat dan memberi score.")
        """
    ),
    md(
        """
        ## Next steps

        Setelah notebook berhasil dijalankan:

        1. Review `baseline_model_comparison.csv`, `winner_hyperparameter_tuning.csv`, dan `test_recall_by_typology.csv`.
        2. Presentasikan hasil berdasarkan Average Precision, Recall@Top-1%, Precision@Top-1%, dan rule-miss recovery.
        3. Gunakan `models/aml_anomaly_detection/best_anomaly_model.joblib` di Streamlit untuk memberi ranking pada
           batch transaksi ABT baru.
        4. Pada tahap hybrid berikutnya, gabungkan `any_rule_alert` dan `anomaly_score` **setelah** evaluasi model murni,
           bukan sebagai input ML.
        """
    ),
]


notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {
        "display_name": "super",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
    },
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Created {NOTEBOOK_PATH}")
