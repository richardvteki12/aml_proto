import { AppHeader } from "@/components/AppHeader";
import { loadDashboardData } from "@/lib/dashboard";

const integer = new Intl.NumberFormat("id-ID");
const decimal = new Intl.NumberFormat("id-ID", { maximumFractionDigits: 2 });

function percent(value: number) {
  return `${decimal.format(value)}%`;
}

function severityClass(severity: string) {
  return severity.toLowerCase();
}

export default function EvaluationPage() {
  const data = loadDashboardData();
  const generatedAt = new Intl.DateTimeFormat("id-ID", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "Asia/Jakarta",
  }).format(new Date(data.generatedAt));

  return (
    <>
      <AppHeader />
      <main className="page-shell">
        <p className="eyebrow">PAGE 1 · EVALUASI DATA</p>
        <h1 className="page-heading">Rule-based, ML, dan ground truth dalam satu evaluasi.</h1>
        <p className="lede">
          Angka pada halaman ini berasal dari artefak data sintetik yang telah dianalisis. Rule
          adalah red flag untuk review; model ML memberi ranking anomali. Keduanya bukan
          penetapan hukum bahwa suatu transaksi adalah AML.
        </p>

        <section className="section" aria-labelledby="overview-title">
          <h2 id="overview-title" className="section-heading">Jejak data yang dianalisis</h2>
          <div className="metric-grid">
            <article className="metric-card">
              <span className="metric-value">{integer.format(data.overview.transactions)}</span>
              <span className="metric-label">Baris Analytical Base Table</span>
              <span className="metric-caption">{integer.format(data.overview.abtColumns)} kolom ABT</span>
            </article>
            <article className="metric-card">
              <span className="metric-value">{integer.format(data.overview.customers)}</span>
              <span className="metric-label">Customers</span>
              <span className="metric-caption">{integer.format(data.overview.accounts)} accounts</span>
            </article>
            <article className="metric-card">
              <span className="metric-value">{integer.format(data.overview.counterparties)}</span>
              <span className="metric-label">Counterparties</span>
              <span className="metric-caption">Data mentah sebelum jahit ABT</span>
            </article>
            <article className="metric-card">
              <span className="metric-value">{integer.format(data.groundTruth.activeScopeTransactions)}</span>
              <span className="metric-label">Ground truth dalam scope aktif</span>
              <span className="metric-caption">dari {integer.format(data.groundTruth.allInjectedTransactions)} injeksi AML</span>
            </article>
          </div>
        </section>

        <section className="section two-column" aria-labelledby="ground-truth-title">
          <article className="panel">
            <div className="panel-header">
              <h2 id="ground-truth-title" className="panel-title">Ground truth sintetik</h2>
              <p className="panel-subtitle">
                Sepuluh skenario disuntikkan generator. Lima skenario pertama adalah scope rule
                dan label ML pada versi ini.
              </p>
            </div>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Scenario</th>
                    <th>Tipologi</th>
                    <th className="numeric">Transaksi</th>
                    <th>Scope sekarang</th>
                  </tr>
                </thead>
                <tbody>
                  {data.groundTruth.scenarios.map((scenario) => (
                    <tr key={scenario.scenarioId}>
                      <td className="mono">{scenario.scenarioId}</td>
                      <td>{scenario.name}</td>
                      <td className="numeric">{integer.format(scenario.transactions)}</td>
                      <td>
                        <span className={`badge ${scenario.inActiveScope ? "low" : ""}`}>
                          {scenario.inActiveScope ? "Rule + ML" : "Generator saja"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>

          <aside className="notice">
            <strong>Batas evaluasi penting.</strong><br />
            {integer.format(data.groundTruth.activeScopeTransactions)} transaksi dari AML-S01 s.d.
            AML-S05 dipakai untuk rule dan evaluasi ML. AML-S06 s.d. AML-S10 tetap ada pada
            generator/ground truth, tetapi belum menjadi target model saat ini; karena itu tidak
            boleh dicampur ke denominator metrik ML.
          </aside>
        </section>

        <section className="section" aria-labelledby="rule-title">
          <h2 id="rule-title" className="section-heading">Rule-based evaluation</h2>
          <div className="metric-grid">
            <article className="metric-card">
              <span className="metric-value">{integer.format(data.rules.anyRuleCandidateHits)}</span>
              <span className="metric-label">Candidate red flags</span>
              <span className="metric-caption">{percent(data.rules.candidateRatePct)} dari seluruh ABT</span>
            </article>
            <article className="metric-card">
              <span className="metric-value">{percent(data.rules.activeScopeRecallPct)}</span>
              <span className="metric-label">Recall rule pada scope aktif</span>
              <span className="metric-caption">{integer.format(data.rules.activeScopeHits)} / {integer.format(data.groundTruth.activeScopeTransactions)}</span>
            </article>
            <article className="metric-card">
              <span className="metric-value">{integer.format(data.rules.allGroundTruthHits)}</span>
              <span className="metric-label">Ground truth tersentuh rule</span>
              <span className="metric-caption">Mencakup 10 skenario generator</span>
            </article>
            <article className="metric-card">
              <span className="metric-value">5</span>
              <span className="metric-label">Rule aktif</span>
              <span className="metric-caption">Red flag yang transparan dan dapat dijelaskan</span>
            </article>
          </div>
          <div className="panel" style={{ marginTop: 24 }}>
            <div className="panel-header">
              <h3 className="panel-title">Hit dan recall per rule</h3>
              <p className="panel-subtitle">Jumlah hit seluruh ABT dibandingkan dengan recall pada tipologi ground truth yang sesuai.</p>
            </div>
            <div className="panel-body bar-list">
              {data.rules.items.map((rule) => (
                <div className="bar-row" key={rule.id}>
                  <div className="bar-label">
                    <span className="mono">{rule.id}</span> {rule.name} <span className={`badge ${severityClass(rule.severity)}`}>{rule.severity}</span>
                  </div>
                  <div className="bar-track" aria-label={`${rule.name} recall ${rule.recallPct}%`}>
                    <div className="bar-fill" style={{ width: `${Math.max(rule.recallPct, 1)}%` }} />
                  </div>
                  <div className="bar-value">{integer.format(rule.candidateHits)} hit · {percent(rule.recallPct)}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="section" aria-labelledby="ml-title">
          <h2 id="ml-title" className="section-heading">ML anomaly evaluation</h2>
          <div className="two-column">
            <article className="panel">
              <div className="panel-header">
                <h3 className="panel-title">Model terpilih: {data.ml.modelName}</h3>
                <p className="panel-subtitle">
                  {data.ml.rawFeatureColumns} raw feature ABT; kebijakan review adalah top {data.ml.reviewTopFraction * 100}% per batch, bukan threshold skor statis.
                </p>
              </div>
              <div className="panel-body">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Model (validation)</th>
                      <th className="numeric">ROC-AUC</th>
                      <th className="numeric">AP</th>
                      <th className="numeric">Recall top 1%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.ml.baseline.map((model) => (
                      <tr key={model.modelName}>
                        <td>{model.modelName}</td>
                        <td className="numeric">{decimal.format(model.rocAuc)}</td>
                        <td className="numeric">{decimal.format(model.averagePrecision)}</td>
                        <td className="numeric">{percent(model.recallAtTopK * 100)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
            <aside className="notice">
              <strong>Kenapa LOF dipilih?</strong><br />
              Pada validation, Local Outlier Factor memberi Average Precision lebih tinggi daripada
              Isolation Forest. Hyperparameter kemudian dicari hanya di validation; headline di
              bawah memakai holdout test akhir yang terpisah.
            </aside>
          </div>

          <div className="metric-grid" style={{ marginTop: 24 }}>
            <article className="metric-card">
              <span className="metric-value">{decimal.format(data.ml.finalTest.rocAuc)}</span>
              <span className="metric-label">ROC-AUC · final holdout</span>
            </article>
            <article className="metric-card">
              <span className="metric-value">{decimal.format(data.ml.finalTest.apLift)}×</span>
              <span className="metric-label">Average precision vs random</span>
            </article>
            <article className="metric-card">
              <span className="metric-value">{percent(data.ml.finalTest.recallAtTopKPct)}</span>
              <span className="metric-label">Recall pada top 1% review</span>
              <span className="metric-caption">{integer.format(data.ml.finalTest.topKTruePositives)} / {integer.format(data.ml.finalTest.knownAmlPositives)} known AML</span>
            </article>
            <article className="metric-card">
              <span className="metric-value">{percent(data.hybrid.combinedRecallPct)}</span>
              <span className="metric-label">Recall gabungan rule atau ML</span>
              <span className="metric-caption">Pada holdout population yang sama</span>
            </article>
          </div>

          <div className="panel" style={{ marginTop: 24 }}>
            <div className="panel-header">
              <h3 className="panel-title">Recall ML per tipologi pada final holdout</h3>
              <p className="panel-subtitle">Top 1% berarti {integer.format(data.ml.finalTest.topKRows)} transaksi teratas dari {integer.format(data.ml.finalTest.rows)} transaksi test.</p>
            </div>
            <div className="panel-body bar-list">
              {data.ml.byTypology.map((item) => (
                <div className="bar-row" key={item.scenarioId}>
                  <div className="bar-label"><span className="mono">{item.scenarioId}</span> {item.name}</div>
                  <div className="bar-track" aria-label={`${item.name} recall ${item.recallAtTopKPct}%`}>
                    <div className="bar-fill" style={{ width: `${Math.max(item.recallAtTopKPct, 1)}%` }} />
                  </div>
                  <div className="bar-value">{percent(item.recallAtTopKPct)}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <footer className="evaluation-footer">
          Snapshot dibuat {generatedAt} WIB. {data.sourceNote}
        </footer>
      </main>
    </>
  );
}
