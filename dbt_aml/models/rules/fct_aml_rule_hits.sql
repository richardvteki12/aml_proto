{{
    config(
        materialized='table',
        indexes=[
            {'columns': ['transaction_id', 'rule_id'], 'unique': true},
            {'columns': ['risk_level']},
            {'columns': ['transaction_timestamp']}
        ]
    )
}}

-- Grain: one triggered and enabled AML monitoring rule per transaction.
-- A hit is a red flag requiring review; it is not a confirmed AML case.

with evaluations as (

    select * from {{ ref('transaction_rule_evaluation') }}

),

rule_catalogue as (

    select * from {{ ref('rule_catalogue') }}

),

candidate_hits as (

    select
        evaluations.transaction_id,
        evaluations.transaction_timestamp,
        evaluations.sender_customer_id,
        evaluations.sender_account_id,
        evaluations.receiver_customer_id,
        evaluations.receiver_account_id,
        evaluations.amount_idr_equivalent,

        rule_values.rule_id,
        rule_values.evidence

    from evaluations

    cross join lateral (
        values
            (
                'R01'::text,
                evaluations.rule_r01_large_transaction_triggered,
                jsonb_build_object(
                    'amount_idr_equivalent', evaluations.amount_idr_equivalent,
                    'rule', 'amount above configured threshold'
                )
            ),
            (
                'R02'::text,
                evaluations.rule_r02_cash_transaction_triggered,
                jsonb_build_object(
                    'transaction_type', evaluations.transaction_type,
                    'amount_idr_equivalent', evaluations.amount_idr_equivalent
                )
            ),
            (
                'R03'::text,
                evaluations.rule_r03_structuring_smurfing_triggered,
                jsonb_build_object(
                    'current_amount_idr', evaluations.amount_idr_equivalent,
                    'small_outbound_count_prev_24h',
                    evaluations.sender_small_outbound_count_prev_24h,
                    'small_outbound_amount_sum_prev_24h',
                    evaluations.sender_small_outbound_amount_sum_prev_24h
                )
            ),
            (
                'R04'::text,
                evaluations.rule_r04_rapid_fund_movement_triggered,
                jsonb_build_object(
                    'inbound_amount_prev_24h',
                    evaluations.sender_successful_inbound_amount_sum_prev_24h,
                    'outgoing_to_inflow_ratio',
                    evaluations.outgoing_amount_to_prior_24h_inflow_ratio
                )
            ),
            (
                'R05'::text,
                evaluations.rule_r05_dormant_account_triggered,
                jsonb_build_object(
                    'hours_since_previous_activity',
                    evaluations.sender_hours_since_previous_account_activity,
                    'amount_idr_equivalent', evaluations.amount_idr_equivalent
                )
            ),
            (
                'R06'::text,
                evaluations.rule_r06_high_velocity_triggered,
                jsonb_build_object(
                    'activity_count_prev_1h',
                    evaluations.sender_account_activity_count_prev_1h,
                    'activity_count_prev_24h',
                    evaluations.sender_account_activity_count_prev_24h
                )
            ),
            (
                'R07'::text,
                evaluations.rule_r07_round_amount_triggered,
                jsonb_build_object(
                    'amount_idr_equivalent', evaluations.amount_idr_equivalent,
                    'is_round_amount_multiple_100m',
                    evaluations.is_round_amount_multiple_100m
                )
            ),
            (
                'R08'::text,
                evaluations.rule_r08_transaction_spike_triggered,
                jsonb_build_object(
                    'amount_to_average_ratio',
                    evaluations.transaction_amount_to_sender_avg_prev_30d_ratio,
                    'history_count_prev_30d',
                    evaluations.sender_successful_outbound_count_prev_30d
                )
            ),
            (
                'R09'::text,
                evaluations.rule_r09_high_risk_country_triggered,
                jsonb_build_object(
                    'sender_country', evaluations.sender_country,
                    'receiver_country', evaluations.receiver_country,
                    'destination_country', evaluations.destination_country,
                    'counterparty_country', evaluations.counterparty_country
                )
            ),
            (
                'R10'::text,
                evaluations.rule_r10_sanction_match_triggered,
                jsonb_build_object(
                    'sanctions_exact_candidate_count',
                    evaluations.sanctions_exact_candidate_count,
                    'sanctions_max_risk_score',
                    evaluations.sanctions_max_risk_score,
                    'match_type', 'exact_normalized_name_candidate'
                )
            ),
            (
                'R11'::text,
                evaluations.rule_r11_pep_customer_triggered,
                jsonb_build_object(
                    'sender_is_pep', evaluations.sender_is_pep,
                    'receiver_is_pep', evaluations.receiver_is_pep,
                    'amount_to_income_ratio',
                    evaluations.transaction_amount_to_monthly_income_ratio,
                    'amount_to_average_ratio',
                    evaluations.transaction_amount_to_sender_avg_prev_30d_ratio
                )
            ),
            (
                'R12'::text,
                evaluations.rule_r12_many_beneficiaries_triggered,
                jsonb_build_object(
                    'distinct_beneficiary_count_prev_24h',
                    evaluations.sender_distinct_beneficiary_count_prev_24h
                )
            ),
            (
                'R13'::text,
                evaluations.rule_r13_many_senders_triggered,
                jsonb_build_object(
                    'distinct_inbound_sender_customer_count_prev_24h',
                    evaluations.sender_distinct_inbound_sender_customer_count_prev_24h
                )
            ),
            (
                'R14'::text,
                evaluations.rule_r14_circular_transaction_triggered,
                jsonb_build_object(
                    'prior_reverse_flow_count_prev_7d',
                    evaluations.prior_reverse_flow_count_prev_7d,
                    'prior_reverse_flow_amount_sum_prev_7d',
                    evaluations.prior_reverse_flow_amount_sum_prev_7d
                )
            ),
            (
                'R15'::text,
                evaluations.rule_r15_frequent_international_transfer_triggered,
                jsonb_build_object(
                    'international_outbound_count_prev_7d',
                    evaluations.sender_international_outbound_count_prev_7d,
                    'is_international_transaction',
                    evaluations.is_international_transaction
                )
            ),
            (
                'R16'::text,
                evaluations.rule_r16_unusual_transaction_time_triggered,
                jsonb_build_object(
                    'transaction_hour', evaluations.transaction_hour
                )
            ),
            (
                'R17'::text,
                evaluations.rule_r17_high_risk_merchant_triggered,
                jsonb_build_object(
                    'status', 'not_applicable',
                    'reason', 'merchant risk attributes are unavailable'
                )
            ),
            (
                'R18'::text,
                evaluations.rule_r18_inconsistent_customer_profile_triggered,
                jsonb_build_object(
                    'amount_to_monthly_income_ratio',
                    evaluations.transaction_amount_to_monthly_income_ratio,
                    'monthly_income', evaluations.sender_monthly_income
                )
            ),
            (
                'R19'::text,
                evaluations.rule_r19_multiple_device_ip_triggered,
                jsonb_build_object(
                    'other_customer_count_same_device_prev_30d',
                    evaluations.other_customer_count_same_device_prev_30d,
                    'other_customer_count_same_ip_prev_30d',
                    evaluations.other_customer_count_same_ip_prev_30d
                )
            ),
            (
                'R20'::text,
                evaluations.rule_r20_failed_screening_retry_triggered,
                jsonb_build_object(
                    'status', 'not_applicable',
                    'reason', 'screening failure reason is unavailable',
                    'failed_transaction_retry_proxy_triggered',
                    evaluations.failed_transaction_retry_proxy_triggered
                )
            )
    ) as rule_values(rule_id, is_triggered, evidence)

    where rule_values.is_triggered

)

select
    candidate_hits.transaction_id,
    candidate_hits.transaction_timestamp,
    candidate_hits.sender_customer_id,
    candidate_hits.sender_account_id,
    candidate_hits.receiver_customer_id,
    candidate_hits.receiver_account_id,
    candidate_hits.amount_idr_equivalent,

    candidate_hits.rule_id,
    rule_catalogue.rule_name,
    rule_catalogue.risk_level,
    case rule_catalogue.risk_level
        when 'Critical' then 4
        when 'High' then 3
        when 'Medium' then 2
        when 'Low' then 1
        else 0
    end as risk_score,
    candidate_hits.evidence

from candidate_hits
inner join rule_catalogue
    on candidate_hits.rule_id = rule_catalogue.rule_id
where rule_catalogue.is_enabled::boolean
  and rule_catalogue.is_applicable::boolean
