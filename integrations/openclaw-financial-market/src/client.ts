import type { OpenClawPluginApi } from "openclaw/plugin-sdk/core";
import {
  resolveBaseUrl,
  resolveSymbols,
  resolveTimeoutMs,
  type FullPipelineResponse,
  type HealthResponse,
  type LogsResponse,
  type MarketReportResponse,
  type PipelineCollectionResponse,
  type PredictionHistoryResponse,
  type PredictionRow,
  type RequestOverrides,
  type TrainingRun,
  type BatchResponse,
} from "./shared.js";

type RequestInitLike = {
  method?: string;
  path: string;
  query?: Record<string, string | number | boolean | undefined>;
  body?: unknown;
  overrides?: RequestOverrides;
};

async function requestJson<T>(
  api: OpenClawPluginApi,
  params: RequestInitLike,
): Promise<T> {
  const baseUrl = resolveBaseUrl(api, params.overrides);
  const timeoutMs = resolveTimeoutMs(api, params.overrides);
  const url = new URL(`${baseUrl}${params.path}`);

  for (const [key, value] of Object.entries(params.query ?? {})) {
    if (value === undefined) {
      continue;
    }
    url.searchParams.set(key, String(value));
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      method: params.method ?? "GET",
      headers: params.body === undefined ? undefined : { "content-type": "application/json" },
      body: params.body === undefined ? undefined : JSON.stringify(params.body),
      signal: controller.signal,
    });

    const rawText = await response.text();
    const payload = rawText ? JSON.parse(rawText) : {};
    if (!response.ok) {
      const detail =
        payload && typeof payload === "object" && "detail" in payload
          ? String((payload as { detail: unknown }).detail)
          : `${response.status} ${response.statusText}`;
      throw new Error(`Market API request failed: ${detail}`);
    }
    return payload as T;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Market API request timed out after ${timeoutMs} ms.`);
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export async function getHealth(
  api: OpenClawPluginApi,
  overrides?: RequestOverrides,
): Promise<HealthResponse> {
  return requestJson<HealthResponse>(api, {
    path: "/health",
    overrides,
  });
}

export async function collectMarketData(
  api: OpenClawPluginApi,
  options: {
    symbols?: string[] | string | null;
    period?: string;
    interval?: string;
    overrides?: RequestOverrides;
  },
): Promise<PipelineCollectionResponse> {
  return requestJson<PipelineCollectionResponse>(api, {
    method: "POST",
    path: "/market-data/collect",
    body: {
      symbols: resolveSymbols(api, options.symbols),
      period: options.period,
      interval: options.interval,
    },
    overrides: options.overrides,
  });
}

export async function trainModels(
  api: OpenClawPluginApi,
  options: {
    symbols?: string[] | string | null;
    historyPeriod?: string;
    interval?: string;
    forceRefresh?: boolean;
    overrides?: RequestOverrides;
  },
): Promise<BatchResponse<TrainingRun>> {
  return requestJson<BatchResponse<TrainingRun>>(api, {
    method: "POST",
    path: "/models/train",
    body: {
      symbols: resolveSymbols(api, options.symbols),
      history_period: options.historyPeriod,
      interval: options.interval,
      force_refresh: options.forceRefresh === true,
    },
    overrides: options.overrides,
  });
}

export async function generatePredictions(
  api: OpenClawPluginApi,
  options: {
    symbols?: string[] | string | null;
    interval?: string;
    refreshPeriod?: string;
    forceRefresh?: boolean;
    autoTrain?: boolean;
    overrides?: RequestOverrides;
  },
): Promise<BatchResponse<PredictionRow>> {
  return requestJson<BatchResponse<PredictionRow>>(api, {
    method: "POST",
    path: "/predictions/generate",
    body: {
      symbols: resolveSymbols(api, options.symbols),
      interval: options.interval,
      refresh_period: options.refreshPeriod,
      force_refresh: options.forceRefresh === true,
      auto_train: options.autoTrain !== false,
    },
    overrides: options.overrides,
  });
}

export async function getLatestPredictions(
  api: OpenClawPluginApi,
  options: {
    symbol: string;
    limit?: number;
    overrides?: RequestOverrides;
  },
): Promise<PredictionHistoryResponse> {
  return requestJson<PredictionHistoryResponse>(api, {
    path: "/predictions/latest",
    query: {
      symbol: options.symbol.trim().toUpperCase(),
      limit: options.limit ?? 24,
    },
    overrides: options.overrides,
  });
}

export async function getMarketReportDirect(
  api: OpenClawPluginApi,
  options: {
    symbols?: string[] | string | null;
    overrides?: RequestOverrides;
  },
): Promise<MarketReportResponse> {
  const baseUrl = resolveBaseUrl(api, options.overrides);
  const timeoutMs = resolveTimeoutMs(api, options.overrides);
  const symbols = resolveSymbols(api, options.symbols);
  const url = new URL(`${baseUrl}/reports/market-summary`);
  for (const symbol of symbols) {
    url.searchParams.append("symbols", symbol);
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    const rawText = await response.text();
    const payload = rawText ? JSON.parse(rawText) : {};
    if (!response.ok) {
      const detail =
        payload && typeof payload === "object" && "detail" in payload
          ? String((payload as { detail: unknown }).detail)
          : `${response.status} ${response.statusText}`;
      throw new Error(`Market API request failed: ${detail}`);
    }
    return payload as MarketReportResponse;
  } finally {
    clearTimeout(timeout);
  }
}

export async function getLogs(
  api: OpenClawPluginApi,
  options: {
    lines?: number;
    overrides?: RequestOverrides;
  },
): Promise<LogsResponse> {
  return requestJson<LogsResponse>(api, {
    path: "/logs/recent",
    query: {
      lines: options.lines ?? 100,
    },
    overrides: options.overrides,
  });
}

export async function runFullPipeline(
  api: OpenClawPluginApi,
  options: {
    symbols?: string[] | string | null;
    historyPeriod?: string;
    interval?: string;
    overrides?: RequestOverrides;
  },
): Promise<FullPipelineResponse> {
  return requestJson<FullPipelineResponse>(api, {
    method: "POST",
    path: "/pipeline/full-run",
    body: {
      symbols: resolveSymbols(api, options.symbols),
      history_period: options.historyPeriod,
      interval: options.interval,
    },
    overrides: options.overrides,
  });
}
