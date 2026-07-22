"use client";

import { useMemo, useState } from "react";

import type { DashboardData } from "@/lib/dashboard";

type Props = {
  ruleCoverage: DashboardData["groundTruth"]["ruleCoverage"];
  holdoutHybrid: DashboardData["groundTruth"]["holdoutHybrid"];
};

type Perspective = "rules" | "hybrid";

const PAGE_SIZE = 20;
const integer = new Intl.NumberFormat("id-ID");
const decimal = new Intl.NumberFormat("id-ID", { maximumFractionDigits: 2 });
const currency = new Intl.NumberFormat("id-ID", {
  style: "currency",
  currency: "IDR",
  maximumFractionDigits: 0,
});

const ruleStatusLabel = {
  rule_hit: "Rule hit",
  rule_miss: "Rule miss",
} as const;

const hybridChannelLabel = {
  both: "Rule + ML",
  rule_only: "Rule saja",
  ml_only: "ML saja",
  missed: "Belum tertangkap",
} as const;

function percent(value: number) {
  return `${decimal.format(value)}%`;
}

function compactTimestamp(timestamp: string) {
  return timestamp.replace("T", " ").slice(0, 16);
}

/**
 * This is intentionally a small Client Component inside a Server-rendered page.
 * The JSON snapshot is passed as serializable props; filtering never retrains the
 * model and never changes a rule result.
 */
export function GroundTruthCoverage({ ruleCoverage, holdoutHybrid }: Props) {
  const [perspective, setPerspective] = useState<Perspective>("rules");
  const [scenarioFilter, setScenarioFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);

  const isRulePerspective = perspective === "rules";
  const scenarioOptions = useMemo(() => {
    const rows = isRulePerspective ? ruleCoverage.rows : holdoutHybrid.rows;
    return Array.from(
      new Map(rows.map((row) => [row.scenarioId, row.scenarioName])).entries(),
    ).sort(([first], [second]) => first.localeCompare(second));
  }, [holdoutHybrid.rows, isRulePerspective, ruleCoverage.rows]);

  const filteredRuleRows = useMemo(() => {
    const term = search.trim().toLowerCase();
    return ruleCoverage.rows.filter(
      (row) =>
        (scenarioFilter === "all" || row.scenarioId === scenarioFilter) &&
        (statusFilter === "all" || row.status === statusFilter) &&
        (!term || row.transactionId.toLowerCase().includes(term)),
    );
  }, [ruleCoverage.rows, scenarioFilter, search, statusFilter]);

  const filteredHoldoutRows = useMemo(() => {
    const term = search.trim().toLowerCase();
    return holdoutHybrid.rows.filter(
      (row) =>
        (scenarioFilter === "all" || row.scenarioId === scenarioFilter) &&
        (statusFilter === "all" || row.channel === statusFilter) &&
        (!term || row.transactionId.toLowerCase().includes(term)),
    );
  }, [holdoutHybrid.rows, scenarioFilter, search, statusFilter]);

  const filteredRows = isRulePerspective ? filteredRuleRows : filteredHoldoutRows;
  const totalPages = Math.max(1, Math.ceil(filteredRows.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const startIndex = (currentPage - 1) * PAGE_SIZE;
  const endIndex = Math.min(startIndex + PAGE_SIZE, filteredRows.length);
  const visibleRuleRows = filteredRuleRows.slice(startIndex, endIndex);
  const visibleHoldoutRows = filteredHoldoutRows.slice(startIndex, endIndex);

  function changePerspective(nextPerspective: Perspective) {
    setPerspective(nextPerspective);
    setScenarioFilter("all");
    setStatusFilter("all");
    setSearch("");
    setPage(1);
  }

  function resetPageForFilter(next: () => void) {
    next();
    setPage(1);
  }

  return (
    <section className="section" aria-labelledby="coverage-title" data-testid="ground-truth-coverage">
      <h2 id="coverage-title" className="section-heading">Ground truth coverage</h2>
      <p className="section-intro">
        Ground truth adalah label yang sengaja disuntikkan ke data sintetik untuk mengukur
        cakupan deteksi. Label ini tidak menjadi input saat model anomaly detection dilatih.
      </p>

      <div className="metric-grid">
        <article className="metric-card">
          <span className="metric-value">{integer.format(ruleCoverage.population)}</span>
          <span className="metric-label">Injeksi AML aktif</span>
          <span className="metric-caption">AML-S01 sampai AML-S05 untuk cakupan rule</span>
        </article>
        <article className="metric-card">
          <span className="metric-value">{integer.format(ruleCoverage.ruleCaught)}</span>
          <span className="metric-label">Tertangkap minimal satu rule</span>
          <span className="metric-caption">Transaction-level, bukan sekadar jumlah hit rule</span>
        </article>
        <article className="metric-card">
          <span className="metric-value">{integer.format(ruleCoverage.ruleMissed)}</span>
          <span className="metric-label">Belum tertangkap rule</span>
          <span className="metric-caption">Kasus untuk dianalisis lebih lanjut oleh ML atau investigator</span>
        </article>
        <article className="metric-card">
          <span className="metric-value">{percent(ruleCoverage.recallPct)}</span>
          <span className="metric-label">Recall rule pada 500 injeksi</span>
          <span className="metric-caption">Denominator berbeda dari final ML holdout</span>
        </article>
      </div>

      <div className="coverage-layout">
        <article className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Cakupan rule per tipologi</h3>
            <p className="panel-subtitle">Seluruh 500 injeksi pada scope aktif.</p>
          </div>
          <div className="table-wrap">
            <table className="data-table compact-table">
              <thead>
                <tr>
                  <th>Tipologi</th>
                  <th className="numeric">Ground truth</th>
                  <th className="numeric">Rule hit</th>
                  <th className="numeric">Rule miss</th>
                  <th className="numeric">Recall</th>
                </tr>
              </thead>
              <tbody>
                {ruleCoverage.byScenario.map((item) => (
                  <tr key={item.scenarioId}>
                    <td><span className="mono">{item.scenarioId}</span> {item.scenarioName}</td>
                    <td className="numeric">{integer.format(item.population)}</td>
                    <td className="numeric">{integer.format(item.ruleCaught ?? 0)}</td>
                    <td className="numeric">{integer.format(item.ruleMissed ?? 0)}</td>
                    <td className="numeric">{percent(item.recallPct ?? 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <aside className="notice coverage-notice">
          <strong>Dua denominator, dua pertanyaan.</strong><br />
          <span className="mono">500</span> mengukur seberapa luas red flag menangkap semua injeksi
          AML aktif. <span className="mono">{integer.format(holdoutHybrid.population)}</span> mengukur
          hasil gabungan Rule + ML hanya pada final holdout yang belum dilihat model. Angka ini
          sengaja tidak digabung agar evaluasi ML tetap adil.
        </aside>
      </div>

      <article className="panel" style={{ marginTop: 24 }}>
        <div className="panel-header">
          <h3 className="panel-title">Matriks Rule + ML pada final holdout</h3>
          <p className="panel-subtitle">
            {integer.format(holdoutHybrid.population)} transaksi known AML pada set test akhir.
            Kebijakan ML adalah review skor top 1% per batch.
          </p>
        </div>
        <div className="table-wrap">
          <table className="data-table compact-table">
            <thead>
              <tr>
                <th>Channel deteksi</th>
                <th className="numeric">Kasus</th>
                <th className="numeric">Proporsi holdout</th>
                <th>Makna</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><span className="case-badge both">Rule + ML</span></td>
                <td className="numeric">{integer.format(holdoutHybrid.both)}</td>
                <td className="numeric">{percent((holdoutHybrid.both / holdoutHybrid.population) * 100)}</td>
                <td>Keduanya memberi sinyal untuk transaksi yang sama.</td>
              </tr>
              <tr>
                <td><span className="case-badge rule-only">Rule saja</span></td>
                <td className="numeric">{integer.format(holdoutHybrid.ruleOnly)}</td>
                <td className="numeric">{percent((holdoutHybrid.ruleOnly / holdoutHybrid.population) * 100)}</td>
                <td>Pola eksplisit tertangkap rule, tetapi tidak masuk antrean top 1% ML.</td>
              </tr>
              <tr>
                <td><span className="case-badge ml-only">ML saja</span></td>
                <td className="numeric">{integer.format(holdoutHybrid.mlOnly)}</td>
                <td className="numeric">{percent((holdoutHybrid.mlOnly / holdoutHybrid.population) * 100)}</td>
                <td>Nilai tambah ML: kasus lolos rule tetapi naik ke ranking anomali teratas.</td>
              </tr>
              <tr>
                <td><span className="case-badge missed">Belum tertangkap</span></td>
                <td className="numeric">{integer.format(holdoutHybrid.missed)}</td>
                <td className="numeric">{percent((holdoutHybrid.missed / holdoutHybrid.population) * 100)}</td>
                <td>Belum diprioritaskan oleh dua lapisan pada kebijakan threshold saat ini.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>

      <article className="panel" style={{ marginTop: 24 }}>
        <div className="panel-header">
          <h3 className="panel-title">Daftar kasus ground truth</h3>
          <p className="panel-subtitle">
            Filter dan buka kasus secara transaction-level. Data yang ditampilkan sengaja tidak
            menyertakan nama, alamat, atau identifier nasabah.
          </p>
        </div>
        <div className="panel-body">
          <div className="coverage-switch" role="group" aria-label="Pilih perspektif coverage">
            <button
              className={`coverage-tab ${isRulePerspective ? "active" : ""}`}
              type="button"
              onClick={() => changePerspective("rules")}
              aria-pressed={isRulePerspective}
              data-testid="coverage-view-rules"
            >
              Rule coverage ({integer.format(ruleCoverage.population)})
            </button>
            <button
              className={`coverage-tab ${!isRulePerspective ? "active" : ""}`}
              type="button"
              onClick={() => changePerspective("hybrid")}
              aria-pressed={!isRulePerspective}
              data-testid="coverage-view-hybrid"
            >
              Final holdout Rule + ML ({integer.format(holdoutHybrid.population)})
            </button>
          </div>

          <div className="coverage-filters">
            <label className="field">
              <span>Tipologi</span>
              <select
                value={scenarioFilter}
                onChange={(event) => resetPageForFilter(() => setScenarioFilter(event.target.value))}
                data-testid="coverage-scenario-filter"
              >
                <option value="all">Semua tipologi</option>
                {scenarioOptions.map(([scenarioId, scenarioName]) => (
                  <option value={scenarioId} key={scenarioId}>{scenarioId} - {scenarioName}</option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>{isRulePerspective ? "Status rule" : "Channel deteksi"}</span>
              <select
                value={statusFilter}
                onChange={(event) => resetPageForFilter(() => setStatusFilter(event.target.value))}
                data-testid="coverage-status-filter"
              >
                <option value="all">Semua status</option>
                {isRulePerspective ? (
                  <>
                    <option value="rule_hit">Rule hit</option>
                    <option value="rule_miss">Rule miss</option>
                  </>
                ) : (
                  <>
                    <option value="both">Rule + ML</option>
                    <option value="rule_only">Rule saja</option>
                    <option value="ml_only">ML saja</option>
                    <option value="missed">Belum tertangkap</option>
                  </>
                )}
              </select>
            </label>
            <label className="field coverage-search">
              <span>Cari transaction ID</span>
              <input
                type="search"
                value={search}
                onChange={(event) => resetPageForFilter(() => setSearch(event.target.value))}
                placeholder="Mis. TXN0000027058"
                data-testid="coverage-search"
              />
            </label>
          </div>

          <div className="table-wrap coverage-table-wrap">
            <table className="data-table" data-testid="ground-truth-table">
              <thead>
                {isRulePerspective ? (
                  <tr>
                    <th>Transaction ID</th>
                    <th>Tipologi</th>
                    <th>Waktu transaksi</th>
                    <th className="numeric">Nominal</th>
                    <th>Rule yang match</th>
                    <th>Status</th>
                  </tr>
                ) : (
                  <tr>
                    <th>Transaction ID</th>
                    <th>Tipologi</th>
                    <th>Waktu transaksi</th>
                    <th className="numeric">Nominal</th>
                    <th className="numeric">Skor ML</th>
                    <th className="numeric">Rank</th>
                    <th>Channel</th>
                  </tr>
                )}
              </thead>
              <tbody>
                {isRulePerspective ? (
                  visibleRuleRows.map((row) => (
                    <tr key={row.transactionId}>
                      <td className="mono">{row.transactionId}</td>
                      <td><span className="mono">{row.scenarioId}</span><br /><span className="muted">{row.scenarioName}</span></td>
                      <td className="mono">{compactTimestamp(row.transactionTimestamp)}</td>
                      <td className="numeric">{currency.format(row.amountIdr)}</td>
                      <td className="mono">{row.ruleHitIds.length ? row.ruleHitIds.join(", ") : "-"}</td>
                      <td><span className={`case-badge ${row.status}`}>{ruleStatusLabel[row.status]}</span></td>
                    </tr>
                  ))
                ) : (
                  visibleHoldoutRows.map((row) => (
                    <tr key={row.transactionId}>
                      <td className="mono">{row.transactionId}</td>
                      <td><span className="mono">{row.scenarioId}</span><br /><span className="muted">{row.scenarioName}</span></td>
                      <td className="mono">{compactTimestamp(row.transactionTimestamp)}</td>
                      <td className="numeric">{currency.format(row.amountIdr)}</td>
                      <td className="numeric">{row.anomalyScore.toFixed(6)}</td>
                      <td className="numeric">{integer.format(row.anomalyRank)}</td>
                      <td><span className={`case-badge ${row.channel}`}>{hybridChannelLabel[row.channel]}</span></td>
                    </tr>
                  ))
                )}
                {filteredRows.length === 0 && (
                  <tr>
                    <td className="empty-state" colSpan={isRulePerspective ? 6 : 7}>
                      Tidak ada kasus yang cocok dengan filter saat ini.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="pagination" aria-label="Pagination daftar ground truth">
            <span className="muted">
              {filteredRows.length
                ? `Menampilkan ${integer.format(startIndex + 1)}-${integer.format(endIndex)} dari ${integer.format(filteredRows.length)} kasus`
                : "0 kasus"}
            </span>
            <div className="pagination-actions">
              <button
                type="button"
                className="pagination-button"
                onClick={() => setPage((current) => Math.max(1, current - 1))}
                disabled={currentPage === 1}
              >
                Sebelumnya
              </button>
              <span className="mono">{currentPage} / {totalPages}</span>
              <button
                type="button"
                className="pagination-button"
                onClick={() => setPage((current) => Math.min(totalPages, current + 1))}
                disabled={currentPage === totalPages}
              >
                Berikutnya
              </button>
            </div>
          </div>
        </div>
      </article>
    </section>
  );
}
