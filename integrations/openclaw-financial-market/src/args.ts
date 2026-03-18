import type { OpenClawPluginApi } from "openclaw/plugin-sdk/core";
import {
  parseBooleanFlag,
  parseNumber,
  resolveDefaultInterval,
  resolveDefaultPeriod,
  resolveDefaultReportRefresh,
  resolveSymbols,
  type RequestOverrides,
} from "./shared.js";

export type ParsedCommandArgs = {
  flags: Record<string, string | boolean>;
  positionals: string[];
};

export function tokenizeArgs(input: string): string[] {
  const tokens: string[] = [];
  const matcher = /"([^"]*)"|'([^']*)'|`([^`]*)`|([^\s]+)/g;
  let match: RegExpExecArray | null = null;
  while ((match = matcher.exec(input)) !== null) {
    tokens.push(match[1] ?? match[2] ?? match[3] ?? match[4] ?? "");
  }
  return tokens;
}

export function parseCommandArgs(input: string): ParsedCommandArgs {
  const flags: Record<string, string | boolean> = {};
  const positionals: string[] = [];
  const tokens = tokenizeArgs(input);

  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (!token.startsWith("--")) {
      positionals.push(token);
      continue;
    }

    const trimmed = token.slice(2);
    const [rawKey, inlineValue] = trimmed.split("=", 2);
    const key = rawKey.trim();
    if (!key) {
      continue;
    }
    if (inlineValue !== undefined) {
      flags[key] = inlineValue;
      continue;
    }

    const next = tokens[index + 1];
    if (next && !next.startsWith("--")) {
      flags[key] = next;
      index += 1;
      continue;
    }
    flags[key] = true;
  }

  return { flags, positionals };
}

export function resolveCommandOverrides(flags: Record<string, string | boolean>): RequestOverrides {
  const overrides: RequestOverrides = {};
  if (typeof flags["base-url"] === "string" && flags["base-url"].trim()) {
    overrides.baseUrl = flags["base-url"].trim();
  }
  if (typeof flags.timeout === "string" || typeof flags.timeout === "number") {
    overrides.timeoutMs = parseNumber(flags.timeout, 180000);
  }
  return overrides;
}

export function resolveCommandSymbols(
  api: OpenClawPluginApi,
  flags: Record<string, string | boolean>,
): string[] {
  const raw = typeof flags.symbols === "string" ? flags.symbols : undefined;
  return resolveSymbols(api, raw);
}

export function resolveCommandPeriod(
  api: OpenClawPluginApi,
  flags: Record<string, string | boolean>,
): string {
  return typeof flags.period === "string" && flags.period.trim()
    ? flags.period.trim()
    : resolveDefaultPeriod(api);
}

export function resolveCommandHistoryPeriod(flags: Record<string, string | boolean>): string {
  return typeof flags["history-period"] === "string" && flags["history-period"].trim()
    ? flags["history-period"].trim()
    : typeof flags.period === "string" && flags.period.trim()
      ? flags.period.trim()
      : "6mo";
}

export function resolveCommandInterval(
  api: OpenClawPluginApi,
  flags: Record<string, string | boolean>,
): string {
  return typeof flags.interval === "string" && flags.interval.trim()
    ? flags.interval.trim()
    : resolveDefaultInterval(api);
}

export function shouldRefreshReport(
  api: OpenClawPluginApi,
  flags: Record<string, string | boolean>,
): boolean {
  return parseBooleanFlag(flags.refresh, resolveDefaultReportRefresh(api));
}

export function shouldTrain(flags: Record<string, string | boolean>): boolean {
  return parseBooleanFlag(flags.train, false);
}

export function shouldJson(flags: Record<string, string | boolean>): boolean {
  return parseBooleanFlag(flags.json, false);
}

export function resolveLines(flags: Record<string, string | boolean>): number {
  return parseNumber(flags.lines, 100);
}
