import type { OpenClawPluginApi } from "openclaw/plugin-sdk/core";

export type FinancialMarketPluginConfig = {
  baseUrl?: string;
  timeoutMs?: number;
  defaultSymbols?: string[];
  defaultPeriod?: string;
  defaultInterval?: string;
  defaultReportRefresh?: boolean;
  delivery?: {
    every?: string;
    cron?: string;
    tz?: string;
    channel?: string;
    to?: string;
    account?: string;
    agent?: string;
    session?: string;
    bestEffortDeliver?: boolean;
  };
};

export type RequestOverrides = {
  baseUrl?: string;
  timeoutMs?: number;
};

export type MarketReportItem = {
  symbol: string;
  current_price: number;
  price_change_24h: number;
  avg_volume_24h: number;
  predicted_price?: number | null;
  confidence_score?: number | null;
  predicted_change_pct?: number | null;
  prediction_timestamp?: string;
  message?: string;
};

export type MarketReportResponse = {
  generated_at: string;
  symbols: string[];
  items: MarketReportItem[];
  message: string;
};

export type PredictionRow = {
  symbol: string;
  prediction_timestamp: string;
  predicted_price: number;
  confidence_score: number;
  model_version: string;
  current_price?: number;
  predicted_change_pct?: number;
  direction?: string;
  message?: string;
};

export type PredictionHistoryResponse = {
  symbol: string;
  predictions: PredictionRow[];
  message: string;
};

export type PipelineCollectionResponse = {
  status?: string;
  rows_collected?: number;
  rows_by_symbol?: Record<string, number>;
  actuals_updated?: number;
  message?: string;
  symbols?: string[];
  period?: string;
  interval?: string;
  timestamp?: string;
};

export type TrainingRun = {
  symbol: string;
  training_rows: number;
  train_rmse: number;
  test_rmse: number;
  train_mae: number;
  test_mae: number;
  model_version: string;
  trained_at: string;
  message: string;
};

export type BatchResponse<T> = {
  symbols: string[];
  completed: T[];
  failed: Array<{ symbol: string; error: string }>;
  message: string;
  timestamp: string;
};

export type LogsResponse = {
  log_file: string | null;
  lines: string[];
  message: string;
};

export type HealthResponse = {
  status: string;
  message: string;
  default_symbols?: string[];
};

export type FullPipelineResponse = {
  symbols: string[];
  data_collection: PipelineCollectionResponse;
  model_training: BatchResponse<TrainingRun>;
  predictions: BatchResponse<PredictionRow>;
  report: MarketReportResponse;
  timestamp: string;
  message: string;
};

export function resolvePluginConfig(api: OpenClawPluginApi): FinancialMarketPluginConfig {
  return (api.pluginConfig ?? {}) as FinancialMarketPluginConfig;
}

export function resolveBaseUrl(
  api: OpenClawPluginApi,
  overrides?: RequestOverrides,
): string {
  const pluginConfig = resolvePluginConfig(api);
  const value = overrides?.baseUrl || pluginConfig.baseUrl || "http://127.0.0.1:8000";
  return value.replace(/\/+$/, "");
}

export function resolveTimeoutMs(
  api: OpenClawPluginApi,
  overrides?: RequestOverrides,
): number {
  const pluginConfig = resolvePluginConfig(api);
  const rawValue = overrides?.timeoutMs ?? pluginConfig.timeoutMs ?? 180000;
  return Number.isFinite(rawValue) && rawValue > 0 ? Math.floor(rawValue) : 180000;
}

export function uniqueSymbols(symbols: Array<string | undefined | null>): string[] {
  const seen = new Set<string>();
  const output: string[] = [];
  for (const symbol of symbols) {
    const cleaned = typeof symbol === "string" ? symbol.trim().toUpperCase() : "";
    if (!cleaned || seen.has(cleaned)) {
      continue;
    }
    seen.add(cleaned);
    output.push(cleaned);
  }
  return output;
}

export function parseSymbolList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return uniqueSymbols(value.map((entry) => String(entry)));
  }
  if (typeof value !== "string") {
    return [];
  }
  return uniqueSymbols(value.split(",").map((entry) => entry.trim()));
}

export function resolveSymbols(
  api: OpenClawPluginApi,
  input?: string[] | string | null,
): string[] {
  const pluginConfig = resolvePluginConfig(api);
  const parsed =
    typeof input === "string"
      ? parseSymbolList(input)
      : Array.isArray(input)
        ? uniqueSymbols(input)
        : [];
  if (parsed.length > 0) {
    return parsed;
  }
  return uniqueSymbols(pluginConfig.defaultSymbols ?? ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]);
}

export function resolveDefaultPeriod(api: OpenClawPluginApi): string {
  return resolvePluginConfig(api).defaultPeriod || "5d";
}

export function resolveDefaultInterval(api: OpenClawPluginApi): string {
  return resolvePluginConfig(api).defaultInterval || "1h";
}

export function resolveDefaultReportRefresh(api: OpenClawPluginApi): boolean {
  return resolvePluginConfig(api).defaultReportRefresh === true;
}

export function formatJsonText(payload: unknown): string {
  return JSON.stringify(payload, null, 2);
}

export function mergePredictionsIntoReport(
  report: MarketReportResponse,
  predictions?: BatchResponse<PredictionRow>,
): MarketReportResponse {
  if (!predictions || predictions.completed.length === 0) {
    return report;
  }

  const predictionMap = new Map(
    predictions.completed.map((prediction) => [prediction.symbol.trim().toUpperCase(), prediction]),
  );

  const items = report.items.map((item) => {
    const prediction = predictionMap.get(item.symbol.trim().toUpperCase());
    if (!prediction) {
      return item;
    }

    const currentPrice =
      typeof item.current_price === "number" && Number.isFinite(item.current_price)
        ? item.current_price
        : typeof prediction.current_price === "number" && Number.isFinite(prediction.current_price)
          ? prediction.current_price
          : undefined;

    const predictedChangePct =
      typeof currentPrice === "number" && currentPrice !== 0
        ? ((prediction.predicted_price - currentPrice) / currentPrice) * 100
        : prediction.predicted_change_pct;

    return {
      ...item,
      current_price: currentPrice ?? item.current_price,
      prediction_timestamp: prediction.prediction_timestamp,
      predicted_price: prediction.predicted_price,
      confidence_score: prediction.confidence_score,
      predicted_change_pct: predictedChangePct,
      message: prediction.message ?? item.message,
    };
  });

  return {
    ...report,
    items,
  };
}

export function textReply(text: string) {
  return { text };
}

export function jsonToolResult(text: string, details: unknown) {
  return {
    content: [{ type: "text" as const, text }],
    details,
  };
}

export function parseBooleanFlag(value: unknown, defaultValue: boolean = false): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(normalized)) {
      return true;
    }
    if (["0", "false", "no", "off"].includes(normalized)) {
      return false;
    }
  }
  return defaultValue;
}

export function parseNumber(value: unknown, fallback: number): number {
  const parsed =
    typeof value === "number" ? value : typeof value === "string" ? Number.parseInt(value, 10) : NaN;
  return Number.isFinite(parsed) ? parsed : fallback;
}
