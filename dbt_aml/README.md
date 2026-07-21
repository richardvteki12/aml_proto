# AML dbt project

This project transforms the PostgreSQL raw schema loaded from the synthetic CSV files.

## Layer contract

1. raw: Python loads the CSV files here exactly as supplied. No dbt model writes to this schema.
2. staging: dbt views named stg_*. Each view cleans types, standardizes text, and turns structural marker strings into SQL NULL.
3. intermediate and mart: these will be added later for leakage-safe feature windows, the analytical base table, and AML rules.

The conversion of markers happens only in staging. For example, an external recipient has no internal bank customer ID, so EXTERNAL_NOT_BANK_CUSTOMER becomes NULL in stg_transactions.receiver_customer_id. The original marker remains available unchanged in raw.transactions.

## One-time setup

Install the adapter in the same Python environment used to run dbt:

~~~powershell
python -m pip install dbt-postgres
~~~

Copy profiles.yml.example to %USERPROFILE%/.dbt/profiles.yml, then set the password in the current PowerShell session:

~~~powershell
$env:DBT_POSTGRES_PASSWORD = [System.Net.NetworkCredential]::new('', (Read-Host -AsSecureString)).Password
~~~

The command asks for the password without writing it into a project file. Never commit the real profiles.yml.

## Run

From this folder:

~~~powershell
dbt debug
dbt build --select staging
~~~

dbt build creates five views in the PostgreSQL staging schema and executes the model tests in models/staging/stg_schema.yml.
