"use client";

import { FormEvent, useState } from "react";
import { AppHeader } from "@/components/AppHeader";

type Scalar = string | number | boolean;
type FormState = Record<string, Scalar>;

type RuleResult = {
  id: string;
  name: string;
  severity: string;
  hit: boolean;
  reason: string;
};

type InferenceResult = {
  modelUsed: string;
  modelScopeWarning: string | null;
  rules: RuleResult[];
  ruleHitCount: number;
  ml: null | {
    modelName: string;
    score: number;
    referencePercentile: number;
    band: string;
    explanation: string;
    reviewPolicy: string;
    referenceDistribution: Record<string, number>;
  };
};

const categoryOptions: Record<string, string[]> = {
  transaction_type: ["BI-FAST", "Cash", "RTGS", "SWIFT", "Transfer"],
  channel: ["API", "ATM", "Branch", "Internet", "Mobile"],
  currency: ["EUR", "IDR", "SGD", "USD"],
  purpose_code: ["BILL", "FAMILY", "INVESTMENT", "OTHER", "SALARY", "TRADE"],
  source_of_fund: ["Business", "Investment", "Salary", "Unknown"],
  destination_country: ["AE", "AU", "GB", "ID", "JP", "KP", "MY", "RU", "SG", "SY", "US"],
  sender_customer_segment: ["Corporate", "Priority", "Retail", "SME"],
  sender_customer_risk_rating: ["High", "Low", "Medium"],
  sender_account_type: ["Business", "Current", "Saving"],
  sender_account_risk_level: ["High", "Low", "Medium"],
  receiver_party_country: ["AE", "AU", "GB", "ID", "JP", "MY", "SG", "US"],
  receiver_party_risk_level: ["High", "Low", "Medium"],
};

const fieldLabels: Record<string, string> = {
  transaction_type: "Jenis transaksi",
  channel: "Kanal transaksi",
  currency: "Mata uang transaksi",
  purpose_code: "Kode tujuan transaksi",
  source_of_fund: "Sumber dana",
  destination_country: "Negara tujuan",
  sender_customer_segment: "Segmen nasabah pengirim",
  sender_customer_risk_rating: "Risk rating nasabah pengirim",
  sender_account_type: "Jenis rekening pengirim",
  sender_account_risk_level: "Risk level rekening pengirim",
  receiver_party_country: "Negara penerima",
  receiver_party_risk_level: "Risk level penerima",
  has_sufficient_history_30d: "Riwayat 30 hari memadai",
  has_prior_successful_sender_activity: "Ada aktivitas pengirim sebelumnya",
  sender_customer_pep_flag: "Nasabah pengirim PEP",
  is_internal_receiver: "Penerima nasabah internal",
};

const normal: FormState = {
  amount_idr_equivalent: 1_450_000,
  sender_customer_monthly_income: 6_000_000,
  sender_success_txn_count_1h: 1,
  sender_success_amount_sum_1h_idr: 1_450_000,
  sender_success_txn_count_120m: 1,
  sender_success_amount_sum_120m_idr: 1_450_000,
  sender_success_txn_count_24h: 2,
  sender_success_amount_sum_24h_idr: 3_200_000,
  sender_success_txn_count_7d: 8,
  sender_success_amount_sum_7d_idr: 10_500_000,
  sender_subthreshold_txn_count_24h: 1,
  sender_subthreshold_amount_sum_24h_idr: 1_450_000,
  minutes_since_last_internal_inbound: -1,
  outbound_to_last_inbound_ratio: 0,
  prior_success_txn_count_30d: 8,
  amount_to_prior_median_ratio_30d: 0.9,
  days_since_prior_successful_sender_activity: 4,
  receiver_txn_count_24h: 1,
  distinct_senders_to_receiver_24h: 1,
  receiver_amount_sum_24h_idr: 1_450_000,
  receiver_txn_count_7d: 3,
  distinct_senders_to_receiver_7d: 2,
  has_prior_internal_inbound_24h: 0,
  has_sufficient_history_30d: 1,
  has_prior_successful_sender_activity: 1,
  sender_customer_pep_flag: 0,
  is_internal_receiver: false,
  transaction_hour: 14,
  transaction_type: "Transfer",
  channel: "Mobile",
  currency: "IDR",
  purpose_code: "FAMILY",
  source_of_fund: "Salary",
  destination_country: "ID",
  sender_customer_segment: "Retail",
  sender_customer_risk_rating: "Low",
  sender_account_type: "Saving",
  sender_account_risk_level: "Low",
  receiver_party_country: "ID",
  receiver_party_risk_level: "Low",
  is_success: true,
  last_internal_inbound_amount_idr: 0,
  prior_amount_max_30d_idr: 2_000_000,
};

// A preset fills the complete model contract.  The user can still edit every
// editable field afterwards; the preset is a starting point, not a verdict.
const presets: Record<string, { label: string; values: FormState }> = {
  normal: { label: "Transaksi normal", values: normal },
  structuring: {
    label: "Structuring / Smurfing",
    values: {
      ...normal,
      amount_idr_equivalent: 9_500_000,
      sender_success_txn_count_120m: 4,
      sender_success_amount_sum_120m_idr: 38_000_000,
      sender_success_txn_count_24h: 4,
      sender_success_amount_sum_24h_idr: 38_000_000,
      sender_success_txn_count_7d: 6,
      sender_success_amount_sum_7d_idr: 41_000_000,
      sender_subthreshold_txn_count_24h: 4,
      sender_subthreshold_amount_sum_24h_idr: 38_000_000,
      prior_success_txn_count_30d: 8,
      amount_to_prior_median_ratio_30d: 1.7,
      transaction_hour: 22,
      transaction_type: "Cash",
      channel: "Branch",
      purpose_code: "TRADE",
      source_of_fund: "Business",
      sender_customer_segment: "Corporate",
      receiver_party_country: "AU",
      destination_country: "AU",
      prior_amount_max_30d_idr: 9_800_000,
    },
  },
  spike: {
    label: "Sudden Transaction Spike",
    values: {
      ...normal,
      amount_idr_equivalent: 150_000_000,
      sender_success_txn_count_1h: 1,
      sender_success_amount_sum_1h_idr: 150_000_000,
      sender_success_txn_count_120m: 1,
      sender_success_amount_sum_120m_idr: 150_000_000,
      sender_success_txn_count_24h: 1,
      sender_success_amount_sum_24h_idr: 150_000_000,
      sender_success_txn_count_7d: 2,
      sender_success_amount_sum_7d_idr: 152_000_000,
      sender_subthreshold_txn_count_24h: 0,
      sender_subthreshold_amount_sum_24h_idr: 0,
      minutes_since_last_internal_inbound: 635,
      prior_success_txn_count_30d: 6,
      amount_to_prior_median_ratio_30d: 22,
      transaction_hour: 21,
      transaction_type: "RTGS",
      channel: "Mobile",
      is_internal_receiver: true,
      last_internal_inbound_amount_idr: 500_000,
      prior_amount_max_30d_idr: 7_000_000,
    },
  },
  rapid: {
    label: "Rapid Movement of Funds",
    values: {
      ...normal,
      amount_idr_equivalent: 450_000_000,
      sender_success_txn_count_1h: 2,
      sender_success_amount_sum_1h_idr: 940_000_000,
      sender_success_txn_count_120m: 4,
      sender_success_amount_sum_120m_idr: 1_690_000_000,
      sender_success_txn_count_24h: 4,
      sender_success_amount_sum_24h_idr: 1_690_000_000,
      sender_success_txn_count_7d: 5,
      sender_success_amount_sum_7d_idr: 1_698_000_000,
      sender_subthreshold_txn_count_24h: 0,
      sender_subthreshold_amount_sum_24h_idr: 0,
      minutes_since_last_internal_inbound: 5,
      prior_success_txn_count_30d: 5,
      amount_to_prior_median_ratio_30d: 1.3,
      transaction_hour: 23,
      transaction_type: "Transfer",
      channel: "API",
      purpose_code: "TRADE",
      source_of_fund: "Investment",
      last_internal_inbound_amount_idr: 500_000_000,
      prior_amount_max_30d_idr: 490_000_000,
    },
  },
  dormant: {
    label: "Dormant Account Reactivation",
    values: {
      ...normal,
      amount_idr_equivalent: 1_035_000_000,
      sender_success_txn_count_1h: 1,
      sender_success_amount_sum_1h_idr: 1_035_000_000,
      sender_success_txn_count_120m: 1,
      sender_success_amount_sum_120m_idr: 1_035_000_000,
      sender_success_txn_count_24h: 1,
      sender_success_amount_sum_24h_idr: 1_035_000_000,
      sender_success_txn_count_7d: 1,
      sender_success_amount_sum_7d_idr: 1_035_000_000,
      sender_subthreshold_txn_count_24h: 0,
      sender_subthreshold_amount_sum_24h_idr: 0,
      minutes_since_last_internal_inbound: 670,
      prior_success_txn_count_30d: 0,
      amount_to_prior_median_ratio_30d: 0,
      has_sufficient_history_30d: 0,
      days_since_prior_successful_sender_activity: 80,
      transaction_hour: 2,
      transaction_type: "Transfer",
      channel: "ATM",
      purpose_code: "TRADE",
      destination_country: "SG",
      receiver_party_country: "SG",
      last_internal_inbound_amount_idr: 10_700_000,
      prior_amount_max_30d_idr: 0,
    },
  },
  multiple_senders: {
    label: "Multiple Senders → One Receiver",
    values: {
      ...normal,
      amount_idr_equivalent: 55_000_000,
      sender_customer_monthly_income: 31_000_000,
      sender_success_txn_count_24h: 2,
      sender_success_amount_sum_24h_idr: 56_600_000,
      sender_success_txn_count_7d: 2,
      sender_success_amount_sum_7d_idr: 56_600_000,
      sender_subthreshold_txn_count_24h: 0,
      sender_subthreshold_amount_sum_24h_idr: 0,
      prior_success_txn_count_30d: 3,
      amount_to_prior_median_ratio_30d: 13.6,
      receiver_txn_count_24h: 6,
      distinct_senders_to_receiver_24h: 6,
      receiver_amount_sum_24h_idr: 183_000_000,
      receiver_txn_count_7d: 8,
      distinct_senders_to_receiver_7d: 8,
      transaction_hour: 17,
      transaction_type: "Transfer",
      channel: "Mobile",
      purpose_code: "BILL",
      sender_customer_segment: "Priority",
      sender_customer_risk_rating: "Medium",
      sender_account_risk_level: "Medium",
      receiver_party_risk_level: "Medium",
      prior_amount_max_30d_idr: 20_000_000,
    },
  },
};

const numericFields: Array<{ name: string; label: string; hint?: string }> = [
  { name: "amount_idr_equivalent", label: "Nominal ekuivalen IDR" },
  { name: "sender_customer_monthly_income", label: "Pendapatan bulanan pengirim (IDR)" },
  { name: "transaction_hour", label: "Jam transaksi (0–23)" },
];

const activityFields: Array<{ name: string; label: string }> = [
  { name: "sender_success_txn_count_1h", label: "Jumlah transaksi berhasil (1 jam)" },
  { name: "sender_success_amount_sum_1h_idr", label: "Total nominal berhasil (1 jam)" },
  { name: "sender_success_txn_count_120m", label: "Jumlah transaksi berhasil (120 menit)" },
  { name: "sender_success_amount_sum_120m_idr", label: "Total nominal berhasil (120 menit)" },
  { name: "sender_success_txn_count_24h", label: "Jumlah transaksi berhasil (24 jam)" },
  { name: "sender_success_amount_sum_24h_idr", label: "Total nominal berhasil (24 jam)" },
  { name: "sender_success_txn_count_7d", label: "Jumlah transaksi berhasil (7 hari)" },
  { name: "sender_success_amount_sum_7d_idr", label: "Total nominal berhasil (7 hari)" },
  { name: "sender_subthreshold_txn_count_24h", label: "Transaksi sub-threshold (24 jam)" },
  { name: "sender_subthreshold_amount_sum_24h_idr", label: "Total sub-threshold (24 jam)" },
];

const historyFields: Array<{ name: string; label: string; hint?: string }> = [
  { name: "last_internal_inbound_amount_idr", label: "Dana masuk internal terakhir (IDR)", hint: "0 = tidak ada; sistem menurunkan flag dan rasio rapid movement." },
  { name: "minutes_since_last_internal_inbound", label: "Jeda sejak dana masuk internal (menit)", hint: "-1 = tidak ada riwayat." },
  { name: "prior_success_txn_count_30d", label: "Jumlah transaksi historis (30 hari)" },
  { name: "prior_amount_max_30d_idr", label: "Nominal maksimum historis (30 hari)", hint: "Digunakan untuk rule spike." },
  { name: "amount_to_prior_median_ratio_30d", label: "Rasio nominal / median historis (30 hari)", hint: "Input ML." },
  { name: "days_since_prior_successful_sender_activity", label: "Hari sejak aktivitas berhasil sebelumnya", hint: "-1 = tidak ada riwayat." },
];

const receiverFields: Array<{ name: string; label: string }> = [
  { name: "receiver_txn_count_24h", label: "Transaksi ke penerima (24 jam)" },
  { name: "distinct_senders_to_receiver_24h", label: "Pengirim unik ke penerima (24 jam)" },
  { name: "receiver_amount_sum_24h_idr", label: "Total ke penerima (24 jam)" },
  { name: "receiver_txn_count_7d", label: "Transaksi ke penerima (7 hari)" },
  { name: "distinct_senders_to_receiver_7d", label: "Pengirim unik ke penerima (7 hari)" },
];

export default function InferencePage() {
  const [form, setForm] = useState<FormState>(normal);
  const [activePreset, setActivePreset] = useState("normal");
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  function update(name: string, value: Scalar) {
    setActivePreset("custom");
    setForm((current) => ({ ...current, [name]: value }));
  }

  function applyPreset(key: string) {
    setActivePreset(key);
    setForm({ ...presets[key].values });
    setResult(null);
    setError(null);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch("/api/inference", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const body = (await response.json()) as InferenceResult & { error?: string };
      if (!response.ok) {
        throw new Error(body.error ?? "Inference tidak berhasil.");
      }
      setResult(body);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "Inference tidak berhasil.");
    } finally {
      setIsLoading(false);
    }
  }

  function renderNumberField(field: { name: string; label: string; hint?: string }) {
    return (
      <div className="field" key={field.name}>
        <label htmlFor={field.name}>{field.label}</label>
        <input
          id={field.name}
          name={field.name}
          type="number"
          step="any"
          value={String(form[field.name])}
          onChange={(event) => update(field.name, event.target.value)}
        />
        {field.hint ? <span className="field-hint">{field.hint}</span> : null}
      </div>
    );
  }

  return (
    <>
      <AppHeader />
      <main className="page-shell">
        <p className="eyebrow">PAGE 2 · LIVE INFERENCE</p>
        <h1 className="page-heading">Uji transaksi baru dengan rule dan model tersimpan.</h1>
        <p className="lede">
          Pilih preset tipologi untuk mengisi form, atau isi sendiri. Saat dijalankan, aplikasi
          memanggil <span className="mono">best_anomaly_model.joblib</span> yang sudah terlatih;
          model tidak di-fit ulang dari browser maupun server.
        </p>

        <div className="inference-layout">
          <form className="form-card" onSubmit={submit}>
            <p className="section-heading">Preset tipologi</p>
            <div className="preset-group" role="group" aria-label="Preset tipologi AML">
              {Object.entries(presets).map(([key, preset]) => (
                <button
                  key={key}
                  type="button"
                  className={`preset-button ${activePreset === key ? "active" : ""}`}
                  onClick={() => applyPreset(key)}
                >
                  {preset.label}
                </button>
              ))}
            </div>
            <p className="form-disclaimer">
              Preset hanya menyalin nilai awal. Semua input dapat Anda ubah; kandidat rule selalu
              dihitung ulang dari nilai akhir yang dikirim.
            </p>

            <fieldset className="form-section">
              <legend>Transaksi dan profil</legend>
              <div className="form-grid">{numericFields.map(renderNumberField)}</div>
              <div className="form-grid" style={{ marginTop: 15 }}>
                <div className="field">
                  <label htmlFor="is_success">Status transaksi</label>
                  <select id="is_success" value={String(form.is_success)} onChange={(event) => update("is_success", event.target.value === "true")}>
                    <option value="true">Berhasil</option>
                    <option value="false">Gagal / reversed</option>
                  </select>
                  <span className="field-hint">Model saat ini hanya berada dalam scope transaksi berhasil.</span>
                </div>
                {Object.entries(categoryOptions).slice(0, 2).map(([name, options]) => (
                  <div className="field" key={name}>
                    <label htmlFor={name}>{fieldLabels[name]}</label>
                    <select id={name} value={String(form[name])} onChange={(event) => update(name, event.target.value)}>
                      {options.map((option) => <option key={option}>{option}</option>)}
                    </select>
                  </div>
                ))}
              </div>
            </fieldset>

            <fieldset className="form-section">
              <legend>Aktivitas pengirim</legend>
              <div className="form-grid four">{activityFields.map(renderNumberField)}</div>
            </fieldset>

            <fieldset className="form-section">
              <legend>Konteks rapid, spike, dan dormant</legend>
              <div className="form-grid">{historyFields.map(renderNumberField)}</div>
              <div className="form-grid" style={{ marginTop: 15 }}>
                {[
                  "has_sufficient_history_30d",
                  "has_prior_successful_sender_activity",
                  "sender_customer_pep_flag",
                  "is_internal_receiver",
                ].map((name) => (
                  <div className="field" key={name}>
                    <label htmlFor={name}>{fieldLabels[name]}</label>
                    <select id={name} value={String(form[name])} onChange={(event) => update(name, event.target.value === "true")}>
                      <option value="true">Ya / 1</option>
                      <option value="false">Tidak / 0</option>
                    </select>
                  </div>
                ))}
              </div>
              <p className="form-disclaimer" style={{ marginTop: 14 }}>
                Rasio keluar/masuk untuk model dihitung otomatis dari nominal transaksi dan dana masuk internal terakhir: <span className="mono">{Number(form.last_internal_inbound_amount_idr) > 0 ? (Number(form.amount_idr_equivalent) / Number(form.last_internal_inbound_amount_idr)).toFixed(2) : "0.00"}</span>.
              </p>
            </fieldset>

            <fieldset className="form-section">
              <legend>Konteks penerima / fan-in</legend>
              <div className="form-grid">{receiverFields.map(renderNumberField)}</div>
            </fieldset>

            <fieldset className="form-section">
              <legend>Kategori bisnis untuk model</legend>
              <div className="form-grid four">
                {Object.entries(categoryOptions).slice(2).map(([name, options]) => (
                  <div className="field" key={name}>
                    <label htmlFor={name}>{fieldLabels[name]}</label>
                    <select id={name} value={String(form[name])} onChange={(event) => update(name, event.target.value)}>
                      {options.map((option) => <option key={option}>{option}</option>)}
                    </select>
                  </div>
                ))}
              </div>
            </fieldset>

            <div className="submit-row">
              <button className="button button-primary" type="submit" disabled={isLoading} data-testid="run-inference">
                {isLoading ? "Menjalankan model…" : "Jalankan inference"}
              </button>
              <span className="form-disclaimer">40 raw feature dikirim ke bundle model. Ground truth tidak pernah dikirim.</span>
            </div>
            {error ? <p className="error-message" role="alert">{error}</p> : null}
          </form>

          <aside className="result-card" aria-live="polite" data-testid="inference-result">
            {!result ? (
              <div className="result-empty">
                <p className="section-heading">Hasil inference</p>
                Belum ada hasil. Isi atau pilih preset, lalu jalankan inference untuk melihat rule yang
                terpenuhi, skor model, dan alasan masing-masing.
              </div>
            ) : (
              <>
                <div className="result-block">
                  <p className="section-heading">Rule AML monitoring</p>
                  <h2 className={result.ruleHitCount ? "risk-high" : "risk-low"} style={{ margin: 0, fontFamily: "var(--serif)", fontSize: 28, fontWeight: 500 }}>
                    {result.ruleHitCount ? `${result.ruleHitCount} red flag terpicu` : "Tidak ada rule yang terpicu"}
                  </h2>
                </div>
                <div className="result-block">
                  {result.rules.map((rule) => (
                    <div className="rule-result" key={rule.id}>
                      <div className="rule-result-title">
                        <span><span className="mono">{rule.id}</span> {rule.name}</span>
                        <span className={`badge ${rule.hit ? severityClass(rule.severity) : ""}`}>{rule.hit ? "HIT" : "Tidak hit"}</span>
                      </div>
                      <p className="rule-result-reason">{rule.reason}</p>
                    </div>
                  ))}
                </div>
                <div className="result-block">
                  <p className="section-heading">ML anomaly score</p>
                  {result.ml ? (
                    <>
                      <p className="field-hint">Artifact aktif: <span className="mono">{result.modelUsed}</span> · {result.ml.modelName}</p>
                      <div className="score-value">{result.ml.score.toFixed(6)}</div>
                      <p className="score-label"><strong className={result.ml.band.includes("Sangat") ? "risk-high" : result.ml.band.includes("Tidak") ? "risk-medium" : "risk-low"}>{result.ml.band}</strong> · persentil referensi {result.ml.referencePercentile.toFixed(2)}%</p>
                      <p className="rule-result-reason">{result.ml.explanation}</p>
                      <p className="rule-result-reason">{result.ml.reviewPolicy}</p>
                      <p className="field-hint">Referensi holdout: p95 {result.ml.referenceDistribution.p95.toFixed(6)} · p99 {result.ml.referenceDistribution.p99.toFixed(6)}</p>
                    </>
                  ) : <p className="rule-result-reason">{result.modelScopeWarning}</p>}
                </div>
              </>
            )}
          </aside>
        </div>
      </main>
    </>
  );
}

function severityClass(severity: string) {
  return severity.toLowerCase();
}
