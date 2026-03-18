import type { OpenClawPluginApi } from "openclaw/plugin-sdk/core";
import {
  collectMarketData,
  generatePredictions,
  getLogs,
  getMarketReportDirect,
  runFullPipeline,
  trainModels,
} from "./client.js";
import {
  parseCommandArgs,
  resolveCommandHistoryPeriod,
  resolveCommandInterval,
  resolveCommandOverrides,
  resolveCommandPeriod,
  resolveCommandSymbols,
  resolveLines,
  shouldJson,
  shouldRefreshReport,
  shouldTrain,
} from "./args.js";
import {
  formatCollection,
  formatFullPipeline,
  formatLogs,
  formatPredictionBatch,
  formatReport,
  formatTraining,
} from "./format.js";
import { formatJsonText, mergePredictionsIntoReport, textReply } from "./shared.js";

async function buildReportCommandResponse(api: OpenClawPluginApi, argsText: string) {
  const { flags } = parseCommandArgs(argsText);
  const symbols = resolveCommandSymbols(api, flags);
  const overrides = resolveCommandOverrides(flags);
  const period = resolveCommandPeriod(api, flags);
  const historyPeriod = resolveCommandHistoryPeriod(flags);
  const interval = resolveCommandInterval(api, flags);
  const refresh = shouldRefreshReport(api, flags);
  const train = shouldTrain(flags);
  let predictionResponse;

  if (refresh) {
    await collectMarketData(api, {
      symbols,
      period,
      interval,
      overrides,
    });
  }

  if (train) {
    await trainModels(api, {
      symbols,
      historyPeriod,
      interval,
      overrides,
    });
  }

  predictionResponse = await generatePredictions(api, {
    symbols,
    interval,
    refreshPeriod: period,
    forceRefresh: false,
    autoTrain: true,
    overrides,
  });

  const report = await getMarketReportDirect(api, {
    symbols,
    overrides,
  });
  const mergedReport = mergePredictionsIntoReport(report, predictionResponse);
  return shouldJson(flags) ? formatJsonText(mergedReport) : formatReport(mergedReport);
}

async function buildRunCommandResponse(api: OpenClawPluginApi, argsText: string) {
  const { flags } = parseCommandArgs(argsText);
  const response = await runFullPipeline(api, {
    symbols: resolveCommandSymbols(api, flags),
    historyPeriod: resolveCommandHistoryPeriod(flags),
    interval: resolveCommandInterval(api, flags),
    overrides: resolveCommandOverrides(flags),
  });
  return shouldJson(flags) ? formatJsonText(response) : formatFullPipeline(response);
}

async function buildLogsCommandResponse(api: OpenClawPluginApi, argsText: string) {
  const { flags } = parseCommandArgs(argsText);
  const response = await getLogs(api, {
    lines: resolveLines(flags),
    overrides: resolveCommandOverrides(flags),
  });
  return shouldJson(flags) ? formatJsonText(response) : formatLogs(response);
}

export function registerFinancialMarketCommands(api: OpenClawPluginApi) {
  api.registerCommand({
    name: "marketreport",
    description: "Generate a market report from the financial market API.",
    acceptsArgs: true,
    handler: async (ctx) => {
      const text = await buildReportCommandResponse(api, ctx.args?.trim() ?? "");
      return textReply(text);
    },
  });

  api.registerCommand({
    name: "marketrun",
    description: "Run the full market data, training, prediction, and reporting pipeline.",
    acceptsArgs: true,
    handler: async (ctx) => {
      const text = await buildRunCommandResponse(api, ctx.args?.trim() ?? "");
      return textReply(text);
    },
  });

  api.registerCommand({
    name: "marketlogs",
    description: "Show recent service log lines from the financial market API.",
    acceptsArgs: true,
    handler: async (ctx) => {
      const text = await buildLogsCommandResponse(api, ctx.args?.trim() ?? "");
      return textReply(text);
    },
  });
}

export async function runCollectForCli(api: OpenClawPluginApi, args: {
  symbols: string[];
  period: string;
  interval: string;
  overrides?: { baseUrl?: string; timeoutMs?: number };
  json?: boolean;
}) {
  const response = await collectMarketData(api, args);
  return args.json ? formatJsonText(response) : formatCollection(response);
}

export async function runPredictForCli(api: OpenClawPluginApi, args: {
  symbols: string[];
  period: string;
  interval: string;
  overrides?: { baseUrl?: string; timeoutMs?: number };
  json?: boolean;
}) {
  const response = await generatePredictions(api, {
    symbols: args.symbols,
    refreshPeriod: args.period,
    interval: args.interval,
    autoTrain: true,
    overrides: args.overrides,
  });
  return args.json ? formatJsonText(response) : formatPredictionBatch(response);
}

export async function runTrainForCli(api: OpenClawPluginApi, args: {
  symbols: string[];
  period: string;
  interval: string;
  overrides?: { baseUrl?: string; timeoutMs?: number };
  json?: boolean;
}) {
  const response = await trainModels(api, {
    symbols: args.symbols,
    historyPeriod: args.period,
    interval: args.interval,
    overrides: args.overrides,
  });
  return args.json ? formatJsonText(response) : formatTraining(response);
}
