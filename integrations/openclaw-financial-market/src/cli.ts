import { spawnSync } from "node:child_process";
import type { Command } from "commander";
import type { OpenClawConfig, OpenClawPluginApi } from "openclaw/plugin-sdk/core";
import { getHealth, getLogs, getMarketReportDirect } from "./client.js";
import {
  runCollectForCli,
  runPredictForCli,
  runTrainForCli,
} from "./commands.js";
import { formatHealth, formatLogs, formatReport } from "./format.js";
import {
  formatJsonText,
  resolveDefaultInterval,
  resolveDefaultPeriod,
  resolvePluginConfig,
  resolveSymbols,
  type FinancialMarketPluginConfig,
} from "./shared.js";

type LoggerLike = {
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
};

type CliContext = {
  program: Command;
  config: OpenClawConfig;
  logger: LoggerLike;
  workspaceDir?: string;
};

type CommonCliOptions = {
  symbols?: string;
  period?: string;
  historyPeriod?: string;
  interval?: string;
  baseUrl?: string;
  timeout?: string;
  json?: boolean;
};

function resolveCliOverrides(options: { baseUrl?: string; timeout?: string }) {
  return {
    baseUrl: options.baseUrl?.trim() || undefined,
    timeoutMs: options.timeout ? Number.parseInt(options.timeout, 10) : undefined,
  };
}

function resolveCliSymbols(api: OpenClawPluginApi, options: { symbols?: string }) {
  return resolveSymbols(api, options.symbols?.trim() || undefined);
}

function renderAndPrint(output: string) {
  // eslint-disable-next-line no-console
  console.log(output);
}

function buildCommandMessage(options: {
  symbols?: string[];
  period?: string;
  historyPeriod?: string;
  interval?: string;
  refresh?: boolean;
  train?: boolean;
  baseUrl?: string;
  timeout?: string;
  json?: boolean;
}) {
  const parts = ["/marketreport"];
  if (options.symbols && options.symbols.length > 0) {
    parts.push("--symbols", options.symbols.join(","));
  }
  if (options.period) {
    parts.push("--period", options.period);
  }
  if (options.historyPeriod) {
    parts.push("--history-period", options.historyPeriod);
  }
  if (options.interval) {
    parts.push("--interval", options.interval);
  }
  if (options.refresh) {
    parts.push("--refresh");
  }
  if (options.train) {
    parts.push("--train");
  }
  if (options.baseUrl) {
    parts.push("--base-url", options.baseUrl);
  }
  if (options.timeout) {
    parts.push("--timeout", options.timeout);
  }
  if (options.json) {
    parts.push("--json");
  }
  return parts.join(" ");
}

function resolveDeliveryDefaults(pluginConfig: FinancialMarketPluginConfig) {
  return pluginConfig.delivery ?? {};
}

function buildCronArgs(params: {
  pluginConfig: FinancialMarketPluginConfig;
  name?: string;
  description?: string;
  every?: string;
  cron?: string;
  at?: string;
  tz?: string;
  channel?: string;
  to?: string;
  account?: string;
  agent?: string;
  session?: string;
  refresh?: boolean;
  train?: boolean;
  symbols?: string[];
  period?: string;
  interval?: string;
  historyPeriod?: string;
  baseUrl?: string;
  timeout?: string;
  bestEffortDeliver?: boolean;
  disabled?: boolean;
  json?: boolean;
}) {
  const defaults = resolveDeliveryDefaults(params.pluginConfig);
  const args = ["cron", "add"];
  const name = params.name?.trim() || "financial-market-report";
  args.push("--name", name);

  if (params.description?.trim()) {
    args.push("--description", params.description.trim());
  }

  const every = params.every?.trim() || defaults.every?.trim();
  const cron = params.cron?.trim() || defaults.cron?.trim();
  const at = params.at?.trim();
  if (at) {
    args.push("--at", at);
  } else if (cron) {
    args.push("--cron", cron);
  } else if (every) {
    args.push("--every", every);
  } else {
    throw new Error("A schedule is required. Provide --every, --cron, or --at.");
  }

  const tz = params.tz?.trim() || defaults.tz?.trim();
  if (tz) {
    args.push("--tz", tz);
  }

  const message = buildCommandMessage({
    symbols: params.symbols,
    period: params.period,
    historyPeriod: params.historyPeriod,
    interval: params.interval,
    refresh: params.refresh,
    train: params.train,
    baseUrl: params.baseUrl,
    timeout: params.timeout,
    json: false,
  });
  args.push("--message", message);

  const channel = params.channel?.trim() || defaults.channel?.trim();
  const to = params.to?.trim() || defaults.to?.trim();
  const account = params.account?.trim() || defaults.account?.trim();
  const agent = params.agent?.trim() || defaults.agent?.trim();
  const session = params.session?.trim() || defaults.session?.trim() || "isolated";

  if (!channel || !to) {
    throw new Error("Delivery requires both --channel and --to, either directly or in plugin config.");
  }
  if (session !== "isolated") {
    throw new Error(
      "Scheduled market reports must use an isolated session so OpenClaw can run and deliver the report.",
    );
  }

  args.push("--announce", "--channel", channel, "--to", to);

  if (account) {
    args.push("--account", account);
  }
  if (agent) {
    args.push("--agent", agent);
  }
  if (session) {
    args.push("--session", session);
  }

  const bestEffort =
    params.bestEffortDeliver === true ||
    (params.bestEffortDeliver === undefined && defaults.bestEffortDeliver === true);
  if (bestEffort) {
    args.push("--best-effort-deliver");
  }
  if (params.disabled) {
    args.push("--disabled");
  }
  if (params.json) {
    args.push("--json");
  }
  return { args, message };
}

function runNestedOpenClaw(args: string[]) {
  const result = spawnSync("openclaw", args, {
    env: process.env,
    encoding: "utf-8",
  });
  if (result.status !== 0) {
    const stderr = result.stderr?.trim() || result.stdout?.trim() || "OpenClaw command failed.";
    throw new Error(stderr);
  }
  return (result.stdout || "").trim();
}

export function registerFinancialMarketCli(api: OpenClawPluginApi, ctx: CliContext) {
  const root = ctx.program
    .command("financial-market")
    .description("Financial market tools backed by the local FastAPI service.")
    .addHelpText(
      "after",
      () =>
        "\nExamples:\n" +
        "  openclaw financial-market health\n" +
        "  openclaw financial-market report --symbols AAPL,MSFT --refresh\n" +
        "  openclaw financial-market schedule-report --every 1h --channel slack --to C0123456789\n",
    );

  root
    .command("health")
    .description("Check the market API health.")
    .option("--base-url <url>", "Override the API base URL")
    .option("--timeout <ms>", "Override the HTTP timeout in milliseconds")
    .option("--json", "Print raw JSON", false)
    .action(async (options: { baseUrl?: string; timeout?: string; json?: boolean }) => {
      const response = await getHealth(api, resolveCliOverrides(options));
      renderAndPrint(options.json ? formatJsonText(response) : formatHealth(response));
    });

  root
    .command("collect")
    .description("Collect market data into the FastAPI service.")
    .option("--symbols <csv>", "Comma-separated symbol list")
    .option("--period <value>", "Collection period", resolveDefaultPeriod(api))
    .option("--interval <value>", "Collection interval", resolveDefaultInterval(api))
    .option("--base-url <url>", "Override the API base URL")
    .option("--timeout <ms>", "Override the HTTP timeout in milliseconds")
    .option("--json", "Print raw JSON", false)
    .action(async (options: CommonCliOptions) => {
      const output = await runCollectForCli(api, {
        symbols: resolveCliSymbols(api, options),
        period: options.period?.trim() || resolveDefaultPeriod(api),
        interval: options.interval?.trim() || resolveDefaultInterval(api),
        overrides: resolveCliOverrides(options),
        json: options.json === true,
      });
      renderAndPrint(output);
    });

  root
    .command("train")
    .description("Train forecasting models.")
    .option("--symbols <csv>", "Comma-separated symbol list")
    .option("--period <value>", "History period", "6mo")
    .option("--interval <value>", "Training interval", resolveDefaultInterval(api))
    .option("--base-url <url>", "Override the API base URL")
    .option("--timeout <ms>", "Override the HTTP timeout in milliseconds")
    .option("--json", "Print raw JSON", false)
    .action(async (options: CommonCliOptions) => {
      const output = await runTrainForCli(api, {
        symbols: resolveCliSymbols(api, options),
        period: options.period?.trim() || "6mo",
        interval: options.interval?.trim() || resolveDefaultInterval(api),
        overrides: resolveCliOverrides(options),
        json: options.json === true,
      });
      renderAndPrint(output);
    });

  root
    .command("predict")
    .description("Generate fresh price predictions.")
    .option("--symbols <csv>", "Comma-separated symbol list")
    .option("--period <value>", "Refresh period", resolveDefaultPeriod(api))
    .option("--interval <value>", "Prediction interval", resolveDefaultInterval(api))
    .option("--base-url <url>", "Override the API base URL")
    .option("--timeout <ms>", "Override the HTTP timeout in milliseconds")
    .option("--json", "Print raw JSON", false)
    .action(async (options: CommonCliOptions) => {
      const output = await runPredictForCli(api, {
        symbols: resolveCliSymbols(api, options),
        period: options.period?.trim() || resolveDefaultPeriod(api),
        interval: options.interval?.trim() || resolveDefaultInterval(api),
        overrides: resolveCliOverrides(options),
        json: options.json === true,
      });
      renderAndPrint(output);
    });

  root
    .command("report")
    .description("Build a market report. Use --refresh to update data first.")
    .option("--symbols <csv>", "Comma-separated symbol list")
    .option("--period <value>", "Refresh period", resolveDefaultPeriod(api))
    .option("--history-period <value>", "Training history period when --train is enabled", "6mo")
    .option("--interval <value>", "Prediction interval", resolveDefaultInterval(api))
    .option("--refresh", "Refresh data and predictions before reporting", false)
    .option("--train", "Train models before reporting", false)
    .option("--base-url <url>", "Override the API base URL")
    .option("--timeout <ms>", "Override the HTTP timeout in milliseconds")
    .option("--json", "Print raw JSON", false)
    .action(
      async (options: CommonCliOptions & { refresh?: boolean; train?: boolean }) => {
        const symbols = resolveCliSymbols(api, options);
        const period = options.period?.trim() || resolveDefaultPeriod(api);
        const historyPeriod =
          typeof options.historyPeriod === "string" && options.historyPeriod.trim()
            ? options.historyPeriod.trim()
            : "6mo";
        const interval = options.interval?.trim() || resolveDefaultInterval(api);
        const overrides = resolveCliOverrides(options);
        const refresh = options.refresh === true;
        const train = options.train === true;

        if (refresh) {
          await runCollectForCli(api, {
            symbols,
            period,
            interval,
            overrides,
            json: false,
          });
        }

        if (train) {
          await runTrainForCli(api, {
            symbols,
            period: historyPeriod,
            interval,
            overrides,
            json: false,
          });
        }

        if (refresh || train) {
          await runPredictForCli(api, {
            symbols,
            period,
            interval,
            overrides,
            json: false,
          });
        }

        const response = await getMarketReportDirect(api, { symbols, overrides });
        renderAndPrint(options.json ? formatJsonText(response) : formatReport(response));
      },
    );

  root
    .command("logs")
    .description("Show recent market service logs.")
    .option("--lines <n>", "Number of lines to return", "100")
    .option("--base-url <url>", "Override the API base URL")
    .option("--timeout <ms>", "Override the HTTP timeout in milliseconds")
    .option("--json", "Print raw JSON", false)
    .action(async (options: { lines?: string; baseUrl?: string; timeout?: string; json?: boolean }) => {
      const response = await getLogs(api, {
        lines: options.lines ? Number.parseInt(options.lines, 10) : 100,
        overrides: resolveCliOverrides(options),
      });
      renderAndPrint(options.json ? formatJsonText(response) : formatLogs(response));
    });

  root
    .command("schedule-report")
    .description("Create an OpenClaw cron job that delivers /marketreport output to a chat.")
    .option("--name <value>", "Job name", "financial-market-report")
    .option("--description <value>", "Job description")
    .option("--every <value>", "Run every duration, for example 1h")
    .option("--cron <value>", "Cron expression")
    .option("--at <value>", "Run once at an ISO time or +duration")
    .option("--tz <value>", "Timezone for cron expressions")
    .option("--channel <value>", "Delivery channel")
    .option("--to <value>", "Delivery destination")
    .option("--account <value>", "Delivery account id")
    .option("--agent <value>", "Agent id")
    .option("--session <value>", "Session target", "isolated")
    .option("--symbols <csv>", "Comma-separated symbol list")
    .option("--period <value>", "Refresh period", resolveDefaultPeriod(api))
    .option("--history-period <value>", "Training history period when --train is enabled", "6mo")
    .option("--interval <value>", "Prediction interval", resolveDefaultInterval(api))
    .option("--refresh", "Refresh data and predictions before each report", true)
    .option("--train", "Train models before each report", false)
    .option("--base-url <url>", "Override the API base URL used by /marketreport")
    .option("--timeout <ms>", "HTTP timeout used by /marketreport")
    .option("--best-effort-deliver", "Do not fail the job if delivery fails", false)
    .option("--disabled", "Create the cron job disabled", false)
    .option("--dry-run", "Print the generated cron command without executing it", false)
    .option("--json", "Print raw JSON", false)
    .action(
      async (options: {
        name?: string;
        description?: string;
        every?: string;
        cron?: string;
        at?: string;
        tz?: string;
        channel?: string;
        to?: string;
        account?: string;
        agent?: string;
        session?: string;
        symbols?: string;
        period?: string;
        historyPeriod?: string;
        interval?: string;
        refresh?: boolean;
        train?: boolean;
        baseUrl?: string;
        timeout?: string;
        bestEffortDeliver?: boolean;
        disabled?: boolean;
        dryRun?: boolean;
        json?: boolean;
      }) => {
        const pluginConfig = resolvePluginConfig(api);
        const symbols = resolveCliSymbols(api, options);
        const { args, message } = buildCronArgs({
          pluginConfig,
          name: options.name,
          description: options.description,
          every: options.every,
          cron: options.cron,
          at: options.at,
          tz: options.tz,
          channel: options.channel,
          to: options.to,
          account: options.account,
          agent: options.agent,
          session: options.session,
          refresh: options.refresh !== false,
          train: options.train === true,
          symbols,
          period: options.period?.trim() || resolveDefaultPeriod(api),
          historyPeriod: options.historyPeriod?.trim() || "6mo",
          interval: options.interval?.trim() || resolveDefaultInterval(api),
          baseUrl: options.baseUrl?.trim(),
          timeout: options.timeout?.trim(),
          bestEffortDeliver: options.bestEffortDeliver,
          disabled: options.disabled,
          json: options.json,
        });

        if (options.dryRun) {
          const payload = {
            command: ["openclaw", ...args],
            message,
          };
          renderAndPrint(options.json ? formatJsonText(payload) : payload.command.join(" "));
          return;
        }

        const output = runNestedOpenClaw(args);
        renderAndPrint(output || "Cron job created.");
      },
    );
}
