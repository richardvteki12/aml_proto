{{
    config(
        materialized='table',
        indexes=[
            {'columns': ['transaction_id'], 'unique': true},
            {'columns': ['transaction_timestamp']},
            {'columns': ['sender_customer_id']},
            {'columns': ['sender_account_id']}
        ]
    )
}}

-- Grain: one row per transaction_id.
-- This model evaluates red flags only; it does not produce a final AML decision.
-- Rule thresholds are prototype monitoring parameters, not regulatory thresholds.

{% set structuring_single_transaction_max_idr = var(
    'structuring_single_transaction_max_idr',
    100000000
) %}

with abt as (

    select * from {{ ref('transaction_abt') }}

),

parameters as (

    select
        max(numeric_value::numeric) filter (
            where rule_id = 'R01'
              and parameter_name = 'large_transaction_amount_idr'
        ) as r01_large_transaction_amount_idr,

        max(numeric_value::numeric) filter (
            where rule_id = 'R02'
              and parameter_name = 'cash_transaction_amount_idr'
        ) as r02_cash_transaction_amount_idr,

        max(numeric_value::numeric) filter (
            where rule_id = 'R03'
              and parameter_name = 'minimum_transaction_count'
        ) as r03_minimum_transaction_count,
        max(numeric_value::numeric) filter (
            where rule_id = 'R03'
              and parameter_name = 'minimum_total_amount_idr'
        ) as r03_minimum_total_amount_idr,

        max(numeric_value::numeric) filter (
            where rule_id = 'R04'
              and parameter_name = 'minimum_prior_inflow_amount_idr'
        ) as r04_minimum_prior_inflow_amount_idr,
        max(numeric_value::numeric) filter (
            where rule_id = 'R04'
              and parameter_name = 'minimum_inbound_count'
        ) as r04_minimum_inbound_count,
        max(numeric_value::numeric) filter (
            where rule_id = 'R04'
              and parameter_name = 'outgoing_to_inflow_ratio'
        ) as r04_outgoing_to_inflow_ratio,

        max(numeric_value::numeric) filter (
            where rule_id = 'R05'
              and parameter_name = 'dormant_hours'
        ) as r05_dormant_hours,
        max(numeric_value::numeric) filter (
            where rule_id = 'R05'
              and parameter_name = 'minimum_transaction_amount_idr'
        ) as r05_minimum_transaction_amount_idr,

        max(numeric_value::numeric) filter (
            where rule_id = 'R06'
              and parameter_name = 'maximum_activity_count_1h'
        ) as r06_maximum_activity_count_1h,
        max(numeric_value::numeric) filter (
            where rule_id = 'R06'
              and parameter_name = 'maximum_activity_count_24h'
        ) as r06_maximum_activity_count_24h,

        max(numeric_value::numeric) filter (
            where rule_id = 'R07'
              and parameter_name = 'minimum_round_amount_idr'
        ) as r07_minimum_round_amount_idr,

        max(numeric_value::numeric) filter (
            where rule_id = 'R08'
              and parameter_name = 'minimum_history_count_30d'
        ) as r08_minimum_history_count_30d,
        max(numeric_value::numeric) filter (
            where rule_id = 'R08'
              and parameter_name = 'amount_to_average_ratio'
        ) as r08_amount_to_average_ratio,

        max(numeric_value::numeric) filter (
            where rule_id = 'R10'
              and parameter_name = 'minimum_candidate_count'
        ) as r10_minimum_candidate_count,

        max(numeric_value::numeric) filter (
            where rule_id = 'R11'
              and parameter_name = 'amount_to_income_ratio'
        ) as r11_amount_to_income_ratio,
        max(numeric_value::numeric) filter (
            where rule_id = 'R11'
              and parameter_name = 'amount_to_average_ratio'
        ) as r11_amount_to_average_ratio,

        max(numeric_value::numeric) filter (
            where rule_id = 'R12'
              and parameter_name = 'minimum_beneficiary_count'
        ) as r12_minimum_beneficiary_count,

        max(numeric_value::numeric) filter (
            where rule_id = 'R13'
              and parameter_name = 'minimum_inbound_sender_count'
        ) as r13_minimum_inbound_sender_count,

        max(numeric_value::numeric) filter (
            where rule_id = 'R14'
              and parameter_name = 'minimum_reverse_flow_count'
        ) as r14_minimum_reverse_flow_count,

        max(numeric_value::numeric) filter (
            where rule_id = 'R15'
              and parameter_name = 'minimum_international_count_7d'
        ) as r15_minimum_international_count_7d,

        max(numeric_value::numeric) filter (
            where rule_id = 'R16'
              and parameter_name = 'unusual_hour_start'
        ) as r16_unusual_hour_start,
        max(numeric_value::numeric) filter (
            where rule_id = 'R16'
              and parameter_name = 'unusual_hour_end'
        ) as r16_unusual_hour_end,

        max(numeric_value::numeric) filter (
            where rule_id = 'R18'
              and parameter_name = 'minimum_amount_to_income_ratio'
        ) as r18_minimum_amount_to_income_ratio,

        max(numeric_value::numeric) filter (
            where rule_id = 'R19'
              and parameter_name = 'minimum_other_customer_count'
        ) as r19_minimum_other_customer_count,

        max(numeric_value::numeric) filter (
            where rule_id = 'R20'
              and parameter_name = 'minimum_failed_attempt_count_1h'
        ) as r20_minimum_failed_attempt_count_1h

    from {{ ref('rule_parameters') }}

),

country_flags as (

    select
        abt.transaction_id,
        exists (
            select 1
            from {{ ref('high_risk_countries') }} as countries
            where countries.country_code in (
                abt.sender_country,
                abt.receiver_country,
                abt.destination_country,
                abt.counterparty_country
            )
        ) as has_high_risk_country
    from abt

)

select
    abt.*,
    country_flags.has_high_risk_country,

    case
        when abt.transaction_status = 'success'
         and abt.amount_idr_equivalent >= parameters.r01_large_transaction_amount_idr
            then true
        else false
    end as rule_r01_large_transaction_triggered,

    case
        when abt.transaction_status = 'success'
         and abt.is_cash_transaction
         and abt.amount_idr_equivalent >= parameters.r02_cash_transaction_amount_idr
            then true
        else false
    end as rule_r02_cash_transaction_triggered,

    case
        when abt.transaction_status = 'success'
         and abt.amount_idr_equivalent <= {{ structuring_single_transaction_max_idr }}
         and abt.sender_small_outbound_count_prev_24h + 1
                >= parameters.r03_minimum_transaction_count
         and abt.sender_small_outbound_amount_sum_prev_24h
                + abt.amount_idr_equivalent
                >= parameters.r03_minimum_total_amount_idr
            then true
        else false
    end as rule_r03_structuring_smurfing_triggered,

    case
        when abt.transaction_status = 'success'
         and abt.sender_successful_inbound_count_prev_24h
                >= parameters.r04_minimum_inbound_count
         and abt.sender_successful_inbound_amount_sum_prev_24h
                >= parameters.r04_minimum_prior_inflow_amount_idr
         and coalesce(
                abt.outgoing_amount_to_prior_24h_inflow_ratio,
                0
             ) >= parameters.r04_outgoing_to_inflow_ratio
            then true
        else false
    end as rule_r04_rapid_fund_movement_triggered,

    case
        when abt.transaction_status = 'success'
         and abt.sender_hours_since_previous_account_activity
                >= parameters.r05_dormant_hours
         and abt.amount_idr_equivalent
                >= parameters.r05_minimum_transaction_amount_idr
            then true
        else false
    end as rule_r05_dormant_account_triggered,

    case
        when abt.transaction_status = 'success'
         and (
            abt.sender_account_activity_count_prev_1h + 1
                >= parameters.r06_maximum_activity_count_1h
            or abt.sender_account_activity_count_prev_24h + 1
                >= parameters.r06_maximum_activity_count_24h
         )
            then true
        else false
    end as rule_r06_high_velocity_triggered,

    case
        when abt.transaction_status = 'success'
         and abt.is_round_amount_multiple_100m
         and abt.amount_idr_equivalent
                >= parameters.r07_minimum_round_amount_idr
            then true
        else false
    end as rule_r07_round_amount_triggered,

    case
        when abt.transaction_status = 'success'
         and abt.sender_successful_outbound_count_prev_30d
                >= parameters.r08_minimum_history_count_30d
         and coalesce(
                abt.transaction_amount_to_sender_avg_prev_30d_ratio,
                0
             ) >= parameters.r08_amount_to_average_ratio
            then true
        else false
    end as rule_r08_transaction_spike_triggered,

    case
        when abt.transaction_status = 'success'
         and country_flags.has_high_risk_country
            then true
        else false
    end as rule_r09_high_risk_country_triggered,

    case
        when abt.sanctions_exact_candidate_count
                >= parameters.r10_minimum_candidate_count
            then true
        else false
    end as rule_r10_sanction_match_triggered,

    case
        when abt.transaction_status = 'success'
         and (
            coalesce(abt.sender_is_pep, false)
            or coalesce(abt.receiver_is_pep, false)
         )
         and (
            coalesce(
                abt.transaction_amount_to_monthly_income_ratio,
                0
            ) >= parameters.r11_amount_to_income_ratio
            or coalesce(
                abt.transaction_amount_to_sender_avg_prev_30d_ratio,
                0
            ) >= parameters.r11_amount_to_average_ratio
         )
            then true
        else false
    end as rule_r11_pep_customer_triggered,

    case
        when abt.transaction_status = 'success'
         and abt.sender_distinct_beneficiary_count_prev_24h + 1
                >= parameters.r12_minimum_beneficiary_count
            then true
        else false
    end as rule_r12_many_beneficiaries_triggered,

    case
        when abt.transaction_status = 'success'
         and abt.sender_distinct_inbound_sender_customer_count_prev_24h
                >= parameters.r13_minimum_inbound_sender_count
            then true
        else false
    end as rule_r13_many_senders_triggered,

    case
        when abt.transaction_status = 'success'
         and abt.receiver_account_id is not null
         and abt.prior_reverse_flow_count_prev_7d
                >= parameters.r14_minimum_reverse_flow_count
            then true
        else false
    end as rule_r14_circular_transaction_triggered,

    case
        when abt.transaction_status = 'success'
         and abt.is_international_transaction
         and abt.sender_international_outbound_count_prev_7d + 1
                >= parameters.r15_minimum_international_count_7d
            then true
        else false
    end as rule_r15_frequent_international_transfer_triggered,

    case
        when abt.transaction_status = 'success'
         and abt.transaction_hour between
                parameters.r16_unusual_hour_start
                and parameters.r16_unusual_hour_end
            then true
        else false
    end as rule_r16_unusual_transaction_time_triggered,

    false as rule_r17_high_risk_merchant_triggered,

    case
        when abt.transaction_status = 'success'
         and coalesce(
                abt.transaction_amount_to_monthly_income_ratio,
                0
             ) >= parameters.r18_minimum_amount_to_income_ratio
            then true
        else false
    end as rule_r18_inconsistent_customer_profile_triggered,

    case
        when coalesce(
                abt.other_customer_count_same_device_prev_30d,
                0
             ) >= parameters.r19_minimum_other_customer_count
          or coalesce(
                abt.other_customer_count_same_ip_prev_30d,
                0
             ) >= parameters.r19_minimum_other_customer_count
            then true
        else false
    end as rule_r19_multiple_device_ip_triggered,

    false as rule_r20_failed_screening_retry_triggered,

    case
        when abt.sender_failed_outbound_attempt_count_prev_1h
                + case when abt.transaction_status = 'failed' then 1 else 0 end
                >= parameters.r20_minimum_failed_attempt_count_1h
            then true
        else false
    end as failed_transaction_retry_proxy_triggered

from abt
cross join parameters
left join country_flags
    on abt.transaction_id = country_flags.transaction_id
