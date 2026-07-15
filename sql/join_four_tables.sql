

select
    -- Transaction event: all values remain at their original raw meaning.
    t.transaction_id,
    t.transaction_timestamp,
    t.transaction_type,
    t.channel,
    t.transaction_status,
    t.debit_credit,
    t.amount,
    t.currency,
    t.amount_idr_equivalent,
    t.purpose_code,
    t.purpose_description,
    t.reference_number,
    t.source_of_fund,
    t.destination_bank,
    t.destination_country,
    t.device_id,
    t.ip_address,
    t.latitude,
    t.longitude,

    -- Sender: master values have a transaction-level fallback.
    t.sender_customer_id,
    t.sender_account_id,
    coalesce(sender_customer.full_name, t.sender_name) as sender_customer_name,
    coalesce(sender_customer.address_line_1, t.sender_address) as sender_customer_address,
    coalesce(sender_customer.country, t.sender_country) as sender_customer_country,
    sender_customer.monthly_income as sender_customer_monthly_income,
    sender_customer.occupation as sender_customer_occupation,
    sender_customer.customer_segment as sender_customer_segment,
    sender_customer.customer_risk_rating as sender_customer_risk_rating,
    sender_customer.pep_flag as sender_customer_pep_flag,
    sender_customer.onboarding_date as sender_customer_onboarding_date,
    sender_account.account_type as sender_account_type,
    sender_account.currency as sender_account_currency,
    sender_account.branch_code as sender_branch_code,
    sender_account.opening_date as sender_account_opening_date,
    sender_account.account_status as sender_account_status,
    sender_account.risk_level as sender_account_risk_level,

    -- Receiver: one common party representation, without adding a party-type field.
    coalesce(receiver_customer.customer_id, counterparty.counterparty_id)
        as receiver_party_id,
    coalesce(receiver_customer.full_name, counterparty.counterparty_name, t.receiver_name)
        as receiver_party_name,
    coalesce(receiver_customer.address_line_1, counterparty.address, t.receiver_address)
        as receiver_party_address,
    coalesce(receiver_customer.country, counterparty.country, t.receiver_country)
        as receiver_party_country,
    coalesce(receiver_customer.customer_risk_rating, counterparty.risk_level)
        as receiver_party_risk_level,

    -- Retain original transaction fields for traceability; no source field is overwritten.
    t.receiver_customer_id,
    t.receiver_account_id,
    t.counterparty_id,
    t.beneficiary_name,
    t.beneficiary_address

from raw.transactions as t

left join raw.customers as sender_customer
    on t.sender_customer_id = sender_customer.customer_id

left join raw.accounts as sender_account
    on t.sender_account_id = sender_account.account_id
    and t.sender_customer_id = sender_account.customer_id

left join raw.customers as receiver_customer
    on t.receiver_customer_id = receiver_customer.customer_id

left join raw.counterparties as counterparty
    on t.counterparty_id = counterparty.counterparty_id
