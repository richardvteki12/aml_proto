"""Inject coherent AML behaviour scenarios into synthetic transaction data.

The module is deliberately deterministic when the caller supplies a seeded
``numpy.random.Generator``.  It only changes synthetic in-memory DataFrames;
writing CSV files remains the responsibility of notebook 01.

The five scenarios in ``FOCUS_SCENARIO_IDS`` are designed as behavioural
patterns that can later be tested with leakage-safe features.  The remaining
five scenarios are retained for the original assignment's ground-truth scope.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


SCENARIOS = [
    ("AML-S01", "Structuring / Smurfing", "High"),
    ("AML-S02", "Sudden Transaction Spike", "High"),
    ("AML-S03", "Rapid Movement of Funds", "Critical"),
    ("AML-S04", "Dormant Account Reactivation", "Critical"),
    ("AML-S05", "Multiple Senders to One Receiver", "High"),
    ("AML-S06", "One Sender to Multiple Beneficiaries", "High"),
    ("AML-S07", "Circular Transaction", "Critical"),
    ("AML-S08", "High-Risk Geography", "High"),
    ("AML-S09", "Unusual Transaction Purpose", "High"),
    ("AML-S10", "Potential Mule Account", "Critical"),
]

FOCUS_SCENARIO_IDS = {
    "AML-S01",
    "AML-S02",
    "AML-S03",
    "AML-S04",
    "AML-S05",
}


def default_scenario_parameters(run_scale: str) -> dict[str, float | int]:
    """Return prototype test parameters, not legal/regulatory thresholds."""

    if run_scale not in {"full", "smoke"}:
        raise ValueError("run_scale must be 'full' or 'smoke'.")

    return {
        "group_size": 4,
        "structuring_window_minutes": 120,
        "structuring_single_transaction_max_idr": 10_000_000,
        "structuring_group_total_min_idr": 30_000_000,
        "rapid_window_hours": 24,
        "rapid_outbound_to_inbound_min_ratio": 0.80,
        "rapid_outbound_to_inbound_max_ratio": 1.05,
        "spike_history_days": 30,
        "spike_min_prior_transactions": 3 if run_scale == "full" else 2,
        "spike_min_multiple_of_prior_max": 3.0,
        "spike_min_multiple_of_prior_median": 8.0,
        # Smoke data spans only 60 days, so a 60-day observed inactivity gap
        # cannot exist after allowing room for the scenario transaction.
        "dormancy_gap_days": 60 if run_scale == "full" else 20,
        "multiple_senders_window_minutes": 120,
        "multiple_senders_min_customers": 4,
    }


def _strict_prior_history_stats(
    transactions: pd.DataFrame,
    window_days: int,
) -> pd.DataFrame:
    """Calculate prior-only account statistics for successful outbound events.

    A transaction at the same timestamp is never considered historical.  This
    implements the strict-before-current-event rule needed for leakage safety.
    """

    # Keep calculations in NumPy arrays.  The dataset has 250k events and
    # scalar pandas .at writes inside a per-event loop make Run All needlessly
    # slow on a laptop.
    result_index = transactions.index
    result_position = pd.Series(np.arange(len(transactions)), index=result_index)
    prior_count = np.zeros(len(transactions), dtype=np.int32)
    prior_gap_days = np.full(len(transactions), np.nan, dtype=float)

    successful = transactions.loc[
        transactions["transaction_status"].eq("Success"),
        ["sender_account_id", "transaction_timestamp", "amount_idr_equivalent"],
    ].copy()
    successful["source_position"] = result_position.loc[successful.index].to_numpy()
    successful = successful.sort_values(
        ["sender_account_id", "transaction_timestamp", "source_position"]
    )
    history_window = np.timedelta64(window_days, "D")

    for _, account_events in successful.groupby("sender_account_id", sort=False):
        event_times = account_events["transaction_timestamp"].to_numpy(
            dtype="datetime64[ns]"
        )
        event_amounts = account_events["amount_idr_equivalent"].to_numpy(float)
        source_positions = account_events["source_position"].to_numpy()

        # side='left' excludes every event at the current timestamp, which
        # prevents same-timestamp data from leaking into historical features.
        right = np.searchsorted(event_times, event_times, side="left")
        left = np.searchsorted(event_times, event_times - history_window, side="right")
        counts = right - left
        prior_count[source_positions] = counts

        has_prior_event = right > 0
        if has_prior_event.any():
            prior_gap_days[source_positions[has_prior_event]] = (
                (event_times[has_prior_event] - event_times[right[has_prior_event] - 1])
                / np.timedelta64(1, "D")
            ).astype(float)

    return pd.DataFrame(
        {
            "prior_window_count": prior_count,
            "prior_gap_days": prior_gap_days,
        },
        index=result_index,
    )


def _selected_prior_amount_stats(
    transactions: pd.DataFrame,
    requested_indexes: Iterable[int],
    window_days: int,
) -> pd.DataFrame:
    """Calculate strictly-prior median and maximum only for selected rows."""

    requested_indexes = pd.Index(list(requested_indexes))
    result = pd.DataFrame(
        index=requested_indexes,
        data={"prior_window_median": np.nan, "prior_window_max": np.nan},
    )
    if requested_indexes.empty:
        return result

    result_position = pd.Series(np.arange(len(transactions)), index=transactions.index)
    requested_positions = result_position.loc[requested_indexes].to_numpy()
    requested_mask = np.zeros(len(transactions), dtype=bool)
    requested_mask[requested_positions] = True
    position_to_index = pd.Series(transactions.index.to_numpy(), index=np.arange(len(transactions)))

    successful = transactions.loc[
        transactions["transaction_status"].eq("Success"),
        ["sender_account_id", "transaction_timestamp", "amount_idr_equivalent"],
    ].copy()
    successful["source_position"] = result_position.loc[successful.index].to_numpy()
    successful = successful.sort_values(
        ["sender_account_id", "transaction_timestamp", "source_position"]
    )
    history_window = np.timedelta64(window_days, "D")

    for _, account_events in successful.groupby("sender_account_id", sort=False):
        event_times = account_events["transaction_timestamp"].to_numpy(dtype="datetime64[ns]")
        event_amounts = account_events["amount_idr_equivalent"].to_numpy(float)
        source_positions = account_events["source_position"].to_numpy()
        selected_local_positions = np.flatnonzero(requested_mask[source_positions])

        for position in selected_local_positions:
            current_time = event_times[position]
            right = np.searchsorted(event_times, current_time, side="left")
            left = np.searchsorted(
                event_times,
                current_time - history_window,
                side="right",
            )
            prior_amounts = event_amounts[left:right]
            if len(prior_amounts):
                source_index = position_to_index.at[source_positions[position]]
                result.at[source_index, "prior_window_median"] = float(np.median(prior_amounts))
                result.at[source_index, "prior_window_max"] = float(np.max(prior_amounts))

    return result


def inject_aml_scenarios(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    customers: pd.DataFrame,
    counterparties: pd.DataFrame,
    labels_per_scenario: int,
    rng: np.random.Generator,
    parameters: dict[str, float | int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Inject ten scenarios and return transaction/ground-truth artefacts.

    Returns
    -------
    transactions
        The same transaction table, modified in place and returned for clarity.
    aml_ground_truth
        One label row per suspicious transaction.  For rapid movement, only
        the outbound leg is labelled; the inbound leg is causal context.
    rapid_pair_evidence
        Mapping from each labelled rapid outbound transaction to its earlier
        internal inbound transaction.  It is validation evidence only and is
        not exported as a model feature.
    reserved_indices
        All transaction rows reserved by AML scenarios, including unlabelled
        rapid inbound legs.  Sanctions injection should avoid these rows.
    """

    transactions = transactions.copy()
    group_size = int(parameters["group_size"])
    if labels_per_scenario % group_size:
        raise ValueError("labels_per_scenario must be divisible by group_size.")

    account_customer = accounts.set_index("account_id")["customer_id"]
    customer_master = customers.set_index("customer_id")
    counterparty_master = counterparties.set_index("counterparty_id")
    active_accounts = accounts.loc[accounts["account_status"].ne("Closed")].copy()
    one_account_per_customer = active_accounts.drop_duplicates("customer_id")

    if len(active_accounts) < group_size or len(one_account_per_customer) < group_size:
        raise ValueError("Not enough active accounts to inject scenario groups.")

    reserved = np.zeros(len(transactions), dtype=bool)
    truth_rows: list[dict[str, object]] = []
    rapid_pair_rows: list[dict[str, object]] = []

    timeline_start = transactions["transaction_timestamp"].min() + pd.Timedelta(days=1)
    timeline_end = transactions["transaction_timestamp"].max() - pd.Timedelta(days=1)
    if timeline_end <= timeline_start:
        raise ValueError("Transaction history is too short for scenario injection.")

    def reserve_random(count: int) -> np.ndarray:
        available = np.flatnonzero(~reserved)
        if len(available) < count:
            raise ValueError("Not enough unreserved transactions for scenario injection.")
        indices = rng.choice(available, size=count, replace=False)
        reserved[indices] = True
        return np.asarray(indices, dtype=int)

    def reserve_from_candidates(candidates: Iterable[int], count: int) -> np.ndarray:
        candidates = np.asarray(list(candidates), dtype=int)
        candidates = candidates[~reserved[candidates]]
        if len(candidates) < count:
            raise ValueError(
                f"Only {len(candidates)} eligible rows are available; {count} are required."
            )
        indices = rng.choice(candidates, size=count, replace=False)
        reserved[indices] = True
        return np.asarray(indices, dtype=int)

    def scenario_anchor(window_minutes: int) -> pd.Timestamp:
        max_offset_seconds = int((timeline_end - timeline_start).total_seconds())
        anchor_offset = int(rng.integers(0, max_offset_seconds))
        return timeline_start + pd.Timedelta(seconds=anchor_offset)

    def set_idr_amount(row_index: int, amount_idr: float) -> None:
        amount_idr = round(float(amount_idr), 2)
        transactions.at[row_index, "amount"] = amount_idr
        transactions.at[row_index, "currency"] = "IDR"
        transactions.at[row_index, "amount_idr_equivalent"] = amount_idr

    def set_sender(row_index: int, account_id: str) -> None:
        customer_id = account_customer.at[account_id]
        profile = customer_master.loc[customer_id]
        transactions.at[row_index, "sender_account_id"] = account_id
        transactions.at[row_index, "sender_customer_id"] = customer_id
        transactions.at[row_index, "sender_name"] = profile["full_name"]
        transactions.at[row_index, "sender_address"] = profile["address_line_1"]
        transactions.at[row_index, "sender_country"] = profile["country"]

    def set_internal_receiver(row_index: int, account_id: str) -> None:
        customer_id = account_customer.at[account_id]
        profile = customer_master.loc[customer_id]
        transactions.at[row_index, "receiver_customer_id"] = customer_id
        transactions.at[row_index, "receiver_account_id"] = account_id
        transactions.at[row_index, "receiver_name"] = profile["full_name"]
        transactions.at[row_index, "receiver_address"] = profile["address_line_1"]
        transactions.at[row_index, "receiver_country"] = profile["country"]
        transactions.at[row_index, "beneficiary_name"] = profile["full_name"]
        transactions.at[row_index, "beneficiary_address"] = profile["address_line_1"]
        transactions.at[row_index, "counterparty_id"] = ""
        transactions.at[row_index, "destination_country"] = profile["country"]

    def set_external_receiver(row_index: int, counterparty_id: str) -> None:
        profile = counterparty_master.loc[counterparty_id]
        transactions.at[row_index, "receiver_customer_id"] = ""
        transactions.at[row_index, "receiver_account_id"] = ""
        transactions.at[row_index, "receiver_name"] = profile["counterparty_name"]
        transactions.at[row_index, "receiver_address"] = profile["address"]
        transactions.at[row_index, "receiver_country"] = profile["country"]
        transactions.at[row_index, "beneficiary_name"] = profile["counterparty_name"]
        transactions.at[row_index, "beneficiary_address"] = profile["address"]
        transactions.at[row_index, "counterparty_id"] = counterparty_id
        transactions.at[row_index, "destination_country"] = profile["country"]

    def add_truth(
        row_index: int,
        scenario_id: str,
        scenario_name: str,
        expected_risk: str,
        scenario_group_id: str,
        note: str,
    ) -> None:
        truth_rows.append(
            {
                "transaction_id": transactions.at[row_index, "transaction_id"],
                "customer_id": transactions.at[row_index, "sender_customer_id"],
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "injected_flag": 1,
                "expected_risk": expected_risk,
                "notes": note,
                "scenario_group_id": scenario_group_id,
            }
        )

    scenario_lookup = {scenario_id: (name, risk) for scenario_id, name, risk in SCENARIOS}
    external_counterparty_ids = counterparties["counterparty_id"].to_numpy()
    active_account_ids = active_accounts["account_id"].to_numpy()
    unique_customer_account_ids = one_account_per_customer["account_id"].to_numpy()
    group_count = labels_per_scenario // group_size

    # AML-S01: four near-threshold transactions by one account, within two hours.
    s01_name, s01_risk = scenario_lookup["AML-S01"]
    for group_number in range(1, group_count + 1):
        row_indices = reserve_random(group_size)
        sender_account_id = str(rng.choice(active_account_ids))
        counterparties_for_group = rng.choice(external_counterparty_ids, size=2, replace=False)
        anchor = scenario_anchor(int(parameters["structuring_window_minutes"]))
        offsets = np.sort(rng.integers(0, int(parameters["structuring_window_minutes"]), group_size))
        group_id = f"AML-S01-GRP-{group_number:04d}"

        for position, row_index in enumerate(row_indices):
            set_sender(int(row_index), sender_account_id)
            set_external_receiver(int(row_index), str(counterparties_for_group[position % 2]))
            transactions.at[row_index, "transaction_timestamp"] = anchor + pd.Timedelta(minutes=int(offsets[position]))
            transactions.at[row_index, "transaction_status"] = "Success"
            transactions.at[row_index, "transaction_type"] = "Transfer"
            transactions.at[row_index, "purpose_description"] = "Synthetic sub-threshold transfer"
            set_idr_amount(
                int(row_index),
                rng.uniform(
                    float(parameters["structuring_single_transaction_max_idr"]) * 0.91,
                    float(parameters["structuring_single_transaction_max_idr"]) * 0.985,
                ),
            )
            add_truth(
                int(row_index),
                "AML-S01",
                s01_name,
                s01_risk,
                group_id,
                "Four sub-threshold outbound transactions by one sender within the configured window.",
            )

    # AML-S03: each labelled outbound is preceded by a matched on-us inbound.
    s03_name, s03_risk = scenario_lookup["AML-S03"]
    for group_number in range(1, group_count + 1):
        row_indices = reserve_random(group_size * 2)
        target_account_id = str(rng.choice(active_account_ids))
        source_pool = active_account_ids[active_account_ids != target_account_id]
        group_id = f"AML-S03-GRP-{group_number:04d}"
        anchor = scenario_anchor(int(parameters["rapid_window_hours"]) * 60)

        for position in range(group_size):
            inbound_index = int(row_indices[position * 2])
            outbound_index = int(row_indices[position * 2 + 1])
            source_account_id = str(rng.choice(source_pool))
            counterparty_id = str(rng.choice(external_counterparty_ids))
            inbound_amount = rng.uniform(350_000_000, 500_000_000)
            outbound_amount = inbound_amount * rng.uniform(
                float(parameters["rapid_outbound_to_inbound_min_ratio"]),
                float(parameters["rapid_outbound_to_inbound_max_ratio"]),
            )
            inbound_time = anchor + pd.Timedelta(minutes=position * 40)
            outbound_time = inbound_time + pd.Timedelta(minutes=int(rng.integers(5, 21)))

            set_sender(inbound_index, source_account_id)
            set_internal_receiver(inbound_index, target_account_id)
            transactions.at[inbound_index, "transaction_timestamp"] = inbound_time
            transactions.at[inbound_index, "transaction_status"] = "Success"
            transactions.at[inbound_index, "transaction_type"] = "Transfer"
            transactions.at[inbound_index, "purpose_description"] = "Synthetic internal inbound funding"
            set_idr_amount(inbound_index, inbound_amount)

            set_sender(outbound_index, target_account_id)
            set_external_receiver(outbound_index, counterparty_id)
            transactions.at[outbound_index, "transaction_timestamp"] = outbound_time
            transactions.at[outbound_index, "transaction_status"] = "Success"
            transactions.at[outbound_index, "transaction_type"] = "Transfer"
            transactions.at[outbound_index, "purpose_description"] = "Synthetic rapid pass-through movement"
            set_idr_amount(outbound_index, outbound_amount)
            add_truth(
                outbound_index,
                "AML-S03",
                s03_name,
                s03_risk,
                group_id,
                "Outbound transfer follows a matched internal inbound transfer within the configured window.",
            )
            rapid_pair_rows.append(
                {
                    "scenario_group_id": group_id,
                    "inbound_transaction_id": transactions.at[inbound_index, "transaction_id"],
                    "outbound_transaction_id": transactions.at[outbound_index, "transaction_id"],
                    "target_account_id": target_account_id,
                }
            )

    # AML-S05: four different customers send to one external counterparty within two hours.
    s05_name, s05_risk = scenario_lookup["AML-S05"]
    for group_number in range(1, group_count + 1):
        row_indices = reserve_random(group_size)
        sender_account_ids = rng.choice(
            unique_customer_account_ids,
            size=group_size,
            replace=False,
        )
        counterparty_id = str(rng.choice(external_counterparty_ids))
        anchor = scenario_anchor(int(parameters["multiple_senders_window_minutes"]))
        offsets = np.sort(rng.integers(0, int(parameters["multiple_senders_window_minutes"]), group_size))
        group_id = f"AML-S05-GRP-{group_number:04d}"

        for position, row_index in enumerate(row_indices):
            set_sender(int(row_index), str(sender_account_ids[position]))
            set_external_receiver(int(row_index), counterparty_id)
            transactions.at[row_index, "transaction_timestamp"] = anchor + pd.Timedelta(minutes=int(offsets[position]))
            transactions.at[row_index, "transaction_status"] = "Success"
            transactions.at[row_index, "transaction_type"] = "Transfer"
            transactions.at[row_index, "purpose_description"] = "Synthetic funnel beneficiary transfer"
            set_idr_amount(int(row_index), rng.uniform(5_000_000, 60_000_000))
            add_truth(
                int(row_index),
                "AML-S05",
                s05_name,
                s05_risk,
                group_id,
                "Four distinct internal customers send to one external counterparty within the configured window.",
            )

    # Legacy scenarios retained for the original 10-scenario assignment scope.
    for scenario_id in ["AML-S06", "AML-S07", "AML-S08", "AML-S09", "AML-S10"]:
        scenario_name, expected_risk = scenario_lookup[scenario_id]
        row_indices = reserve_random(labels_per_scenario)
        group_ids = [
            f"{scenario_id}-GRP-{position // group_size + 1:04d}"
            for position in range(labels_per_scenario)
        ]

        if scenario_id == "AML-S06":
            for group_number in range(group_count):
                group_rows = row_indices[group_number * group_size : (group_number + 1) * group_size]
                sender_account_id = str(rng.choice(active_account_ids))
                recipient_ids = rng.choice(external_counterparty_ids, size=group_size, replace=False)
                for row_index, recipient_id in zip(group_rows, recipient_ids, strict=True):
                    set_sender(int(row_index), sender_account_id)
                    set_external_receiver(int(row_index), str(recipient_id))
        elif scenario_id == "AML-S07":
            for row_index in row_indices:
                sender_account_id = str(rng.choice(active_account_ids))
                receiver_candidates = active_account_ids[active_account_ids != sender_account_id]
                set_sender(int(row_index), sender_account_id)
                set_internal_receiver(int(row_index), str(rng.choice(receiver_candidates)))
        elif scenario_id == "AML-S08":
            for row_index in row_indices:
                set_external_receiver(int(row_index), str(rng.choice(external_counterparty_ids)))
                country = str(rng.choice(["RU", "SY", "KP"]))
                transactions.at[row_index, "receiver_country"] = country
                transactions.at[row_index, "destination_country"] = country
        elif scenario_id == "AML-S09":
            for row_index in row_indices:
                transactions.at[row_index, "purpose_code"] = "INDUSTRIAL"
                transactions.at[row_index, "purpose_description"] = "Industrial machinery procurement"
                set_idr_amount(int(row_index), rng.uniform(350_000_000, 900_000_000))
        elif scenario_id == "AML-S10":
            for row_index in row_indices:
                transactions.at[row_index, "purpose_description"] = "Synthetic mule-account movement"
                set_idr_amount(int(row_index), rng.uniform(35_000_000, 180_000_000))

        for position, row_index in enumerate(row_indices):
            transactions.at[row_index, "transaction_status"] = "Success"
            add_truth(
                int(row_index),
                scenario_id,
                scenario_name,
                expected_risk,
                group_ids[position],
                f"Synthetic injection for {scenario_name}.",
            )

    # S02/S04 are selected after temporal scenarios are in place, so their
    # historical conditions are evaluated against the final preceding history.
    history = _strict_prior_history_stats(
        transactions,
        int(parameters["spike_history_days"]),
    )

    s04_name, s04_risk = scenario_lookup["AML-S04"]
    dormant_candidates = history.index[
        transactions["transaction_status"].eq("Success")
        & history["prior_gap_days"].ge(float(parameters["dormancy_gap_days"]))
    ].to_numpy()
    # One reactivation per account keeps the observed inactivity gap unambiguous.
    dormant_candidates = (
        transactions.loc[dormant_candidates, ["sender_account_id", "transaction_timestamp"]]
        .sort_values("transaction_timestamp")
        .groupby("sender_account_id", sort=False)
        .head(1)
        .index.to_numpy()
    )
    dormant_indices = reserve_from_candidates(dormant_candidates, labels_per_scenario)
    dormant_sender_accounts = set(transactions.loc[dormant_indices, "sender_account_id"])
    for position, row_index in enumerate(dormant_indices, start=1):
        transactions.at[row_index, "transaction_status"] = "Success"
        transactions.at[row_index, "purpose_description"] = "Synthetic dormant-account reactivation"
        set_external_receiver(int(row_index), str(rng.choice(external_counterparty_ids)))
        set_idr_amount(int(row_index), rng.uniform(180_000_000, 1_100_000_000))
        add_truth(
            int(row_index),
            "AML-S04",
            s04_name,
            s04_risk,
            f"AML-S04-GRP-{position:04d}",
            "Large outbound transaction occurs after a strictly prior inactivity gap.",
        )

    s02_name, s02_risk = scenario_lookup["AML-S02"]
    spike_candidates = history.index[
        transactions["transaction_status"].eq("Success")
        & history["prior_window_count"].ge(int(parameters["spike_min_prior_transactions"]))
    ].to_numpy()
    # One spike per sender account prevents injected spikes from becoming the
    # prior baseline for one another. Dormant-scenario accounts are excluded so
    # their deliberately large reactivation amount cannot alter the spike test.
    spike_candidates = (
        transactions.loc[spike_candidates, ["sender_account_id", "transaction_timestamp"]]
        .loc[lambda frame: ~frame["sender_account_id"].isin(dormant_sender_accounts)]
        .sort_values("transaction_timestamp")
        .groupby("sender_account_id", sort=False)
        .head(1)
        .index.to_numpy()
    )
    spike_indices = reserve_from_candidates(spike_candidates, labels_per_scenario)
    spike_history = _selected_prior_amount_stats(
        transactions,
        spike_indices,
        int(parameters["spike_history_days"]),
    )
    for position, row_index in enumerate(spike_indices, start=1):
        prior_median = float(spike_history.at[row_index, "prior_window_median"])
        prior_max = float(spike_history.at[row_index, "prior_window_max"])
        transactions.at[row_index, "transaction_status"] = "Success"
        transactions.at[row_index, "purpose_description"] = "Synthetic profile-inconsistent transaction spike"
        set_idr_amount(
            int(row_index),
            max(
                prior_median * float(parameters["spike_min_multiple_of_prior_median"]),
                prior_max * float(parameters["spike_min_multiple_of_prior_max"]),
                150_000_000,
            ),
        )
        add_truth(
            int(row_index),
            "AML-S02",
            s02_name,
            s02_risk,
            f"AML-S02-GRP-{position:04d}",
            "Current amount exceeds the account's strictly prior 30-day amount baseline.",
        )

    aml_ground_truth = pd.DataFrame(truth_rows).sort_values(
        ["scenario_id", "scenario_group_id", "transaction_id"]
    ).reset_index(drop=True)
    rapid_pair_evidence = pd.DataFrame(rapid_pair_rows)
    return transactions, aml_ground_truth, rapid_pair_evidence, np.flatnonzero(reserved)


def validate_focus_scenarios(
    transactions: pd.DataFrame,
    aml_ground_truth: pd.DataFrame,
    rapid_pair_evidence: pd.DataFrame,
    parameters: dict[str, float | int],
) -> pd.DataFrame:
    """Return pass/fail acceptance checks for the five feature-engineering scenarios."""

    labelled = aml_ground_truth.merge(
        transactions,
        on="transaction_id",
        how="left",
        validate="1:1",
    )
    checks: list[dict[str, object]] = []
    group_size = int(parameters["group_size"])

    s01 = labelled.loc[labelled["scenario_id"].eq("AML-S01")].copy()
    s01_groups = s01.groupby("scenario_group_id").agg(
        row_count=("transaction_id", "size"),
        sender_accounts=("sender_account_id", "nunique"),
        earliest=("transaction_timestamp", "min"),
        latest=("transaction_timestamp", "max"),
        total_idr=("amount_idr_equivalent", "sum"),
        max_idr=("amount_idr_equivalent", "max"),
    )
    s01_duration = (s01_groups["latest"] - s01_groups["earliest"]).dt.total_seconds() / 60
    checks.extend(
        [
            {"check": "S01: four transactions per structuring group", "passed": s01_groups["row_count"].eq(group_size).all()},
            {"check": "S01: one sender account per structuring group", "passed": s01_groups["sender_accounts"].eq(1).all()},
            {"check": "S01: transactions fit configured time window", "passed": s01_duration.le(float(parameters["structuring_window_minutes"])).all()},
            {"check": "S01: each transaction stays below prototype threshold", "passed": s01_groups["max_idr"].lt(float(parameters["structuring_single_transaction_max_idr"])).all()},
            {"check": "S01: group total exceeds prototype threshold", "passed": s01_groups["total_idr"].ge(float(parameters["structuring_group_total_min_idr"])).all()},
        ]
    )

    history = _strict_prior_history_stats(transactions, int(parameters["spike_history_days"]))
    transaction_index = transactions.set_index("transaction_id").index
    index_lookup = pd.Series(transactions.index.to_numpy(), index=transaction_index)
    s02_indices = index_lookup.loc[labelled.loc[labelled["scenario_id"].eq("AML-S02"), "transaction_id"]].to_numpy()
    s02_amounts = transactions.loc[s02_indices, "amount_idr_equivalent"].to_numpy(float)
    s02_history = history.loc[s02_indices]
    s02_amount_history = _selected_prior_amount_stats(
        transactions,
        s02_indices,
        int(parameters["spike_history_days"]),
    )
    checks.extend(
        [
            {"check": "S02: spike rows have sufficient prior 30-day history", "passed": s02_history["prior_window_count"].ge(int(parameters["spike_min_prior_transactions"])).all()},
            {"check": "S02: spike exceeds prior maximum by configured multiple", "passed": (s02_amounts >= s02_amount_history.loc[s02_indices, "prior_window_max"].to_numpy(float) * float(parameters["spike_min_multiple_of_prior_max"])).all()},
        ]
    )

    rapid_lookup = transactions.set_index("transaction_id")
    if rapid_pair_evidence.empty:
        rapid_passed = False
    else:
        rapid_pairs = rapid_pair_evidence.copy()
        inbound = rapid_lookup.loc[rapid_pairs["inbound_transaction_id"]].reset_index()
        outbound = rapid_lookup.loc[rapid_pairs["outbound_transaction_id"]].reset_index()
        rapid_minutes = (outbound["transaction_timestamp"].to_numpy() - inbound["transaction_timestamp"].to_numpy()) / np.timedelta64(1, "m")
        rapid_ratio = outbound["amount_idr_equivalent"].to_numpy(float) / inbound["amount_idr_equivalent"].to_numpy(float)
        rapid_passed = bool(
            (outbound["sender_account_id"].to_numpy() == inbound["receiver_account_id"].to_numpy()).all()
            and (rapid_minutes > 0).all()
            and (rapid_minutes <= float(parameters["rapid_window_hours"]) * 60).all()
            and (rapid_ratio >= float(parameters["rapid_outbound_to_inbound_min_ratio"])).all()
            and (rapid_ratio <= float(parameters["rapid_outbound_to_inbound_max_ratio"])).all()
        )
    checks.append({"check": "S03: outbound follows matched internal inbound", "passed": rapid_passed})

    s04_indices = index_lookup.loc[labelled.loc[labelled["scenario_id"].eq("AML-S04"), "transaction_id"]].to_numpy()
    checks.append(
        {
            "check": "S04: reactivation rows have configured prior inactivity gap",
            "passed": history.loc[s04_indices, "prior_gap_days"].ge(float(parameters["dormancy_gap_days"])).all(),
        }
    )

    s05 = labelled.loc[labelled["scenario_id"].eq("AML-S05")].copy()
    s05["receiver_party_id"] = np.where(
        s05["receiver_customer_id"].ne("EXTERNAL_NOT_BANK_CUSTOMER"),
        s05["receiver_customer_id"],
        s05["counterparty_id"],
    )
    s05_groups = s05.groupby("scenario_group_id").agg(
        row_count=("transaction_id", "size"),
        sender_customers=("sender_customer_id", "nunique"),
        receiver_parties=("receiver_party_id", "nunique"),
        earliest=("transaction_timestamp", "min"),
        latest=("transaction_timestamp", "max"),
    )
    s05_duration = (s05_groups["latest"] - s05_groups["earliest"]).dt.total_seconds() / 60
    checks.extend(
        [
            {"check": "S05: four transactions per multiple-sender group", "passed": s05_groups["row_count"].eq(group_size).all()},
            {"check": "S05: group has configured distinct sender customers", "passed": s05_groups["sender_customers"].ge(int(parameters["multiple_senders_min_customers"])).all()},
            {"check": "S05: group has one receiver party", "passed": s05_groups["receiver_parties"].eq(1).all()},
            {"check": "S05: transactions fit configured time window", "passed": s05_duration.le(float(parameters["multiple_senders_window_minutes"])).all()},
        ]
    )

    return pd.DataFrame(checks)
