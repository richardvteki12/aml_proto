import fs from "node:fs";
import path from "node:path";

export type GroundTruthRuleCase = {
  transactionId: string;
  scenarioId: string;
  scenarioName: string;
  transactionTimestamp: string;
  amountIdr: number;
  ruleHitIds: string[];
  status: "rule_hit" | "rule_miss";
};

export type GroundTruthHoldoutCase = {
  transactionId: string;
  scenarioId: string;
  scenarioName: string;
  transactionTimestamp: string;
  amountIdr: number;
  anomalyScore: number;
  anomalyRank: number;
  channel: "both" | "rule_only" | "ml_only" | "missed";
};

type CoverageByScenario = {
  scenarioId: string;
  scenarioName: string;
  population: number;
  ruleCaught?: number;
  ruleMissed?: number;
  recallPct?: number;
  both?: number;
  ruleOnly?: number;
  mlOnly?: number;
  missed?: number;
  combined?: number;
  combinedRecallPct?: number;
};

export type DashboardData = {
  generatedAt: string;
  sourceNote: string;
  overview: {
    customers: number;
    accounts: number;
    counterparties: number;
    transactions: number;
    successfulTransactions: number;
    abtColumns: number;
    allGroundTruthTransactions: number;
    activeScopeGroundTruthTransactions: number;
  };
  groundTruth: {
    allInjectedTransactions: number;
    activeScopeTransactions: number;
    activeScenarios: string[];
    scenarios: Array<{
      scenarioId: string;
      name: string;
      transactions: number;
      customers: number;
      scenarioGroups: number;
      inActiveScope: boolean;
    }>;
    ruleCoverage: {
      population: number;
      ruleCaught: number;
      ruleMissed: number;
      recallPct: number;
      byScenario: CoverageByScenario[];
      rows: GroundTruthRuleCase[];
    };
    holdoutHybrid: {
      population: number;
      both: number;
      ruleOnly: number;
      mlOnly: number;
      missed: number;
      combined: number;
      combinedRecallPct: number;
      byScenario: CoverageByScenario[];
      rows: GroundTruthHoldoutCase[];
    };
  };
  rules: {
    population: number;
    anyRuleCandidateHits: number;
    candidateRatePct: number;
    activeScopeHits: number;
    activeScopeRecallPct: number;
    allGroundTruthHits: number;
    allGroundTruthRecallPct: number;
    items: Array<{
      id: string;
      name: string;
      scenarioId: string;
      severity: string;
      candidateHits: number;
      candidateRatePct: number;
      groundTruthTransactions: number;
      ownTypologyTruePositiveHits: number;
      recallPct: number;
      allGroundTruthHits: number;
    }>;
  };
  ml: {
    modelName: string;
    scope: string;
    reviewTopFraction: number;
    rawFeatureColumns: number;
    selectedParameters: Record<string, string | number>;
    baseline: Array<{
      modelName: string;
      rocAuc: number;
      averagePrecision: number;
      apLift: number;
      precisionAtTopK: number;
      recallAtTopK: number;
      topKRows: number;
      topKTruePositives: number;
      fitSeconds: number;
    }>;
    tunedWinner: {
      modelName: string;
      rocAuc: number;
      averagePrecision: number;
      apLift: number;
      precisionAtTopK: number;
      recallAtTopK: number;
      topKRows: number;
      topKTruePositives: number;
      fitSeconds: number;
    };
    finalTest: {
      rows: number;
      knownAmlPositives: number;
      rocAuc: number;
      averagePrecision: number;
      apLift: number;
      topKRows: number;
      topKTruePositives: number;
      precisionAtTopKPct: number;
      recallAtTopKPct: number;
      ruleMissedKnownAml: number;
      ruleMissRecoveredInTopK: number;
      ruleMissRecoveryPct: number;
    };
    byTypology: Array<{
      scenarioId: string;
      name: string;
      knownAmlRows: number;
      topKHits: number;
      recallAtTopKPct: number;
      meanAnomalyScore: number;
    }>;
  };
  hybrid: {
    holdoutKnownAml: number;
    ruleOnlyCaptured: number;
    mlTopOnePctCaptured: number;
    ruleMissed: number;
    ruleMissRecoveredByMl: number;
    combinedCaptured: number;
    combinedRecallPct: number;
  };
};

export function loadDashboardData(): DashboardData {
  const dashboardPath = path.join(process.cwd(), "public", "dashboard-data.json");
  if (!fs.existsSync(dashboardPath)) {
    throw new Error(
      "dashboard-data.json belum tersedia. Jalankan npm run refresh:dashboard dari folder web.",
    );
  }
  return JSON.parse(fs.readFileSync(dashboardPath, "utf8")) as DashboardData;
}
