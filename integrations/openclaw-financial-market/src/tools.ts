import type { OpenClawPluginApi } from "openclaw/plugin-sdk/core";
import {
  collectMarketData,
  generatePredictions,
  getLogs,
  getMarketReportDirect,
  getLatestPredictions,
  runFullPipeline,
  trainModels,
} from "./client.js";
import {
  formatCollection,
  formatFullPipeline,
  formatLogs,
  formatPredictionBatch,
  formatPredictionHistory,
  formatReport,
  formatTraining,
} from "./format.js";
import {
  jsonToolResult,
  mergePredictionsIntoReport,
  parseBooleanFlag,
  parseNumber,
  resolveDefaultInterval,
  resolveDefaultPeriod,
  resolveSymbols,
} from "./shared.js";

function buildRequestOverrides(params: Record<string, unknown>) {
  const overrides: { baseUrl?: string; timeoutMs?: number } = {};
  if (typeof params.baseUrl === "string" && params.baseUrl.trim()) {
    overrides.baseUrl = params.baseUrl.trim();
  }
  if (typeof params.timeoutMs === "number" || typeof params.timeoutMs === "string") {
    overrides.timeoutMs = parseNumber(params.timeoutMs, 180000);
  }
  return overrides;
}

function withBaseFields() {
  return {
    baseUrl: {
      type: "string",
      description: "Optional API base URL override.",
    },
    timeoutMs: {
      type: "number",
      description: "Optional HTTP timeout override in milliseconds.",
    },
  };
}

export function createFinancialMarketTools(api: OpenClawPluginApi) {
  return [
    {
      name: "financial_market_collect",
      label: "Financial Market Collect",
      description: "Collect market data into the financial market service.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          symbols: {
            type: "array",
            items: { type: "string" },
            description: "Symbols to collect.",
          },
          period: {
            type: "string",
            description: "Collection period such as 5d or 1mo.",
          },
          interval: {
            type: "string",
            description: "Collection interval such as 1h.",
          },
          ...withBaseFields(),
        },
      },
      async execute(_id: string, params: Record<string, unknown>) {
        const response = await collectMarketData(api, {
          symbols: resolveSymbols(api, params.symbols as string[] | undefined),
          period:
            typeof params.period === "string" && params.period.trim()
              ? params.period.trim()
              : resolveDefaultPeriod(api),
          interval:
            typeof params.interval === "string" && params.interval.trim()
              ? params.interval.trim()
              : resolveDefaultInterval(api),
          overrides: buildRequestOverrides(params),
        });
        return jsonToolResult(formatCollection(response), response);
      },
    },
    {
      name: "financial_market_train",
      label: "Financial Market Train",
      description: "Train the forecasting models in the financial market service.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          symbols: {
            type: "array",
            items: { type: "string" },
          },
          historyPeriod: {
            type: "string",
            description: "History period used for training refresh, such as 6mo.",
          },
          interval: {
            type: "string",
            description: "Model interval such as 1h.",
          },
          forceRefresh: {
            type: "boolean",
            description: "Refresh market history before training.",
          },
          ...withBaseFields(),
        },
      },
      async execute(_id: string, params: Record<string, unknown>) {
        const response = await trainModels(api, {
          symbols: resolveSymbols(api, params.symbols as string[] | undefined),
          historyPeriod:
            typeof params.historyPeriod === "string" && params.historyPeriod.trim()
              ? params.historyPeriod.trim()
              : "6mo",
          interval:
            typeof params.interval === "string" && params.interval.trim()
              ? params.interval.trim()
              : resolveDefaultInterval(api),
          forceRefresh: parseBooleanFlag(params.forceRefresh, false),
          overrides: buildRequestOverrides(params),
        });
        return jsonToolResult(formatTraining(response), response);
      },
    },
    {
      name: "financial_market_predict",
      label: "Financial Market Predict",
      description: "Generate fresh price predictions using the financial market service.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          symbols: {
            type: "array",
            items: { type: "string" },
          },
          interval: {
            type: "string",
            description: "Prediction interval such as 1h.",
          },
          refreshPeriod: {
            type: "string",
            description: "Market data refresh period before prediction, such as 5d.",
          },
          forceRefresh: {
            type: "boolean",
          },
          autoTrain: {
            type: "boolean",
          },
          ...withBaseFields(),
        },
      },
      async execute(_id: string, params: Record<string, unknown>) {
        const response = await generatePredictions(api, {
          symbols: resolveSymbols(api, params.symbols as string[] | undefined),
          interval:
            typeof params.interval === "string" && params.interval.trim()
              ? params.interval.trim()
              : resolveDefaultInterval(api),
          refreshPeriod:
            typeof params.refreshPeriod === "string" && params.refreshPeriod.trim()
              ? params.refreshPeriod.trim()
              : resolveDefaultPeriod(api),
          forceRefresh: parseBooleanFlag(params.forceRefresh, false),
          autoTrain: parseBooleanFlag(params.autoTrain, true),
          overrides: buildRequestOverrides(params),
        });
        return jsonToolResult(formatPredictionBatch(response), response);
      },
    },
    {
      name: "financial_market_report",
      label: "Financial Market Report",
      description: "Generate a plain-language market report from the financial market service.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          symbols: {
            type: "array",
            items: { type: "string" },
          },
          refresh: {
            type: "boolean",
            description: "Refresh market data and predictions before building the report.",
          },
          train: {
            type: "boolean",
            description: "Train models before generating the report.",
          },
          period: {
            type: "string",
          },
          historyPeriod: {
            type: "string",
            description: "Training history period used when train is enabled, such as 6mo.",
          },
          interval: {
            type: "string",
          },
          ...withBaseFields(),
        },
      },
      async execute(_id: string, params: Record<string, unknown>) {
        const symbols = resolveSymbols(api, params.symbols as string[] | undefined);
        const period =
          typeof params.period === "string" && params.period.trim()
            ? params.period.trim()
            : resolveDefaultPeriod(api);
        const historyPeriod =
          typeof params.historyPeriod === "string" && params.historyPeriod.trim()
            ? params.historyPeriod.trim()
            : "6mo";
        const interval =
          typeof params.interval === "string" && params.interval.trim()
            ? params.interval.trim()
            : resolveDefaultInterval(api);
        const overrides = buildRequestOverrides(params);
        const refresh = parseBooleanFlag(params.refresh, false);
        const train = parseBooleanFlag(params.train, false);
        let predictionResponse;

        if (refresh) {
          await collectMarketData(api, { symbols, period, interval, overrides });
        }
        if (train) {
          await trainModels(api, { symbols, historyPeriod, interval, overrides });
        }
        predictionResponse = await generatePredictions(api, {
          symbols,
          interval,
          refreshPeriod: period,
          autoTrain: true,
          overrides,
        });

        const response = await getMarketReportDirect(api, { symbols, overrides });
        const mergedReport = mergePredictionsIntoReport(response, predictionResponse);
        return jsonToolResult(formatReport(mergedReport), mergedReport);
      },
    },
    {
      name: "financial_market_prediction_history",
      label: "Financial Market Prediction History",
      description: "Read recent prediction history for a single symbol.",
      parameters: {
        type: "object",
        additionalProperties: false,
        required: ["symbol"],
        properties: {
          symbol: {
            type: "string",
          },
          limit: {
            type: "number",
          },
          ...withBaseFields(),
        },
      },
      async execute(_id: string, params: Record<string, unknown>) {
        const symbol = typeof params.symbol === "string" ? params.symbol.trim().toUpperCase() : "";
        if (!symbol) {
          throw new Error("symbol is required");
        }
        const response = await getLatestPredictions(api, {
          symbol,
          limit: parseNumber(params.limit, 24),
          overrides: buildRequestOverrides(params),
        });
        return jsonToolResult(formatPredictionHistory(response), response);
      },
    },
    {
      name: "financial_market_logs",
      label: "Financial Market Logs",
      description: "Read recent service logs from the financial market API.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          lines: {
            type: "number",
          },
          ...withBaseFields(),
        },
      },
      async execute(_id: string, params: Record<string, unknown>) {
        const response = await getLogs(api, {
          lines: parseNumber(params.lines, 100),
          overrides: buildRequestOverrides(params),
        });
        return jsonToolResult(formatLogs(response), response);
      },
    },
    {
      name: "financial_market_pipeline",
      label: "Financial Market Pipeline",
      description: "Run the full collection, training, prediction, and report pipeline.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          symbols: {
            type: "array",
            items: { type: "string" },
          },
          historyPeriod: {
            type: "string",
          },
          interval: {
            type: "string",
          },
          ...withBaseFields(),
        },
      },
      async execute(_id: string, params: Record<string, unknown>) {
        const response = await runFullPipeline(api, {
          symbols: resolveSymbols(api, params.symbols as string[] | undefined),
          historyPeriod:
            typeof params.historyPeriod === "string" && params.historyPeriod.trim()
              ? params.historyPeriod.trim()
              : "6mo",
          interval:
            typeof params.interval === "string" && params.interval.trim()
              ? params.interval.trim()
              : resolveDefaultInterval(api),
          overrides: buildRequestOverrides(params),
        });
        return jsonToolResult(formatFullPipeline(response), response);
      },
    },
  ];
}
