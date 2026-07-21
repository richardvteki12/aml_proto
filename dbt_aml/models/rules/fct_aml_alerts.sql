{{
    config(
        materialized='table',
        indexes=[
            {'columns': ['transaction_id'], 'unique': true},
            {'columns': ['highest_risk_score']},
            {'columns': ['transaction_timestamp']}
        ]
    )
}}

-- Grain: one alert summary for each transaction with one or more enabled rule hits.

with rule_hits as (

    select * from {{ ref('fct_aml_rule_hits') }}

)

select
    transaction_id,
    min(transaction_timestamp) as transaction_timestamp,
    min(sender_customer_id) as sender_customer_id,
    min(sender_account_id) as sender_account_id,
    min(receiver_customer_id) as receiver_customer_id,
    min(receiver_account_id) as receiver_account_id,
    max(amount_idr_equivalent) as amount_idr_equivalent,

    count(*) as triggered_rule_count,
    max(risk_score) as highest_risk_score,
    case max(risk_score)
        when 4 then 'Critical'
        when 3 then 'High'
        when 2 then 'Medium'
        when 1 then 'Low'
        else 'Unknown'
    end as highest_risk_level,
    sum(risk_score) as total_risk_score,

    string_agg(
        rule_id,
        ', ' order by risk_score desc, rule_id
    ) as triggered_rule_ids,
    string_agg(
        rule_name,
        ' | ' order by risk_score desc, rule_id
    ) as triggered_rule_names,
    jsonb_agg(
        jsonb_build_object(
            'rule_id', rule_id,
            'rule_name', rule_name,
            'risk_level', risk_level,
            'risk_score', risk_score,
            'evidence', evidence
        )
        order by risk_score desc, rule_id
    ) as rule_evidence,

    'open'::text as alert_status,
    'rule_engine'::text as alert_source

from rule_hits
group by transaction_id
