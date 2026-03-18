import type {
  BatchResponse,
  FullPipelineResponse,
  HealthResponse,
  LogsResponse,
  MarketReportItem,
  MarketReportResponse,
  PipelineCollectionResponse,
  PredictionHistoryResponse,
  PredictionRow,
  TrainingRun,
} from "./shared.js";

function formatTimestamp(value?: string): string {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function formatCurrency(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a";
  }
  return `$${value.toFixed(2)}`;
}

function formatPercent(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a";
  }
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function formatConfidence(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a";
  }
  return `${value.toFixed(1)}%`;
}

function formatReportLine(item: MarketReportItem): string {
  const parts = [
    `${item.symbol}: ${formatCurrency(item.current_price)}`,
    `24h ${formatPercent(item.price_change_24h)}`,
  ];
  if (typeof item.predicted_price === "number") {
    parts.push(`next ${formatCurrency(item.predicted_price)}`);
    parts.push(`move ${formatPercent(item.predicted_change_pct ?? null)}`);
    parts.push(`confidence ${formatConfidence(item.confidence_score ?? null)}`);
  } else {
    parts.push("next n/a");
  }
  return parts.join(" | ");
}

export function formatHealth(response: HealthResponse): string {
  const defaults =
    Array.isArray(response.default_symbols) && response.default_symbols.length > 0
      ? response.default_symbols.join(", ")
      : "n/a";
  return [
    response.message,
    `Status: ${response.status}`,
    `Default symbols: ${defaults}`,
  ].join("\n");
}

export function formatCollection(response: PipelineCollectionResponse): string {
  const symbols = Array.isArray(response.symbols) ? response.symbols.join(", ") : "n/a";
  const rowsBySymbol = response.rows_by_symbol
    ? Object.entries(response.rows_by_symbol)
        .map(([symbol, count]) => `${symbol} ${count}`)
        .join(", ")
    : "n/a";
  return [
    response.message || "Market collection completed.",
    `Symbols: ${symbols}`,
    `Rows collected: ${response.rows_collected ?? 0}`,
    `Rows by symbol: ${rowsBySymbol}`,
    `Actuals updated: ${response.actuals_updated ?? 0}`,
  ].join("\n");
}

export function formatTraining(response: BatchResponse<TrainingRun>): string {
  const completed = response.completed
    .map(
      (item) =>
        `${item.symbol}: rows ${item.training_rows}, test RMSE ${item.test_rmse.toFixed(4)}, model ${item.model_version}`,
    )
    .join("\n");
  const failed = response.failed
    .map((item) => `${item.symbol}: ${item.error}`)
    .join("\n");
  return [
    response.message,
    completed ? `Completed:\n${completed}` : "Completed: none",
    failed ? `Failed:\n${failed}` : "Failed: none",
  ].join("\n");
}

export function formatPredictionBatch(response: BatchResponse<PredictionRow>): string {
  const completed = response.completed
    .map(
      (item) =>
        `${item.symbol}: current ${formatCurrency(item.current_price)} | next ${formatCurrency(item.predicted_price)} | move ${formatPercent(item.predicted_change_pct ?? null)} | confidence ${formatConfidence(item.confidence_score)}`,
    )
    .join("\n");
  const failed = response.failed
    .map((item) => `${item.symbol}: ${item.error}`)
    .join("\n");
  return [
    response.message,
    completed ? `Completed:\n${completed}` : "Completed: none",
    failed ? `Failed:\n${failed}` : "Failed: none",
  ].join("\n");
}

export function formatPredictionHistory(response: PredictionHistoryResponse): string {
  if (response.predictions.length === 0) {
    return response.message;
  }
  const lines = response.predictions.map((item) => {
    return `${formatTimestamp(item.prediction_timestamp)} | ${response.symbol} | ${formatCurrency(item.predicted_price)} | confidence ${formatConfidence(item.confidence_score)}`;
  });
  return [response.message, ...lines].join("\n");
}

export function formatReport(response: MarketReportResponse): string {
  if (!response.items || response.items.length === 0) {
    return response.message;
  }
  return [
    `Market report`,
    `Generated: ${formatTimestamp(response.generated_at)}`,
    response.message,
    "",
    ...response.items.map((item) => formatReportLine(item)),
  ].join("\n");
}

export function formatLogs(response: LogsResponse): string {
  const header = response.log_file ? `Log file: ${response.log_file}` : "Log file: unavailable";
  const body = response.lines.length > 0 ? response.lines.join("\n") : "(no log lines available)";
  return [header, body].join("\n");
}

export function formatFullPipeline(response: FullPipelineResponse): string {
  return [
    response.message,
    `Completed at: ${formatTimestamp(response.timestamp)}`,
    "",
    formatCollection(response.data_collection),
    "",
    formatTraining(response.model_training),
    "",
    formatPredictionBatch(response.predictions),
    "",
    formatReport(response.report),
  ].join("\n");
}
