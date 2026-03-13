import type { AnyAgentTool, OpenClawPluginApi } from "openclaw/plugin-sdk/core";
import { registerFinancialMarketCli } from "./src/cli.js";
import { registerFinancialMarketCommands } from "./src/commands.js";
import { createFinancialMarketTools } from "./src/tools.js";

export default function register(api: OpenClawPluginApi) {
  registerFinancialMarketCommands(api);

  for (const tool of createFinancialMarketTools(api)) {
    api.registerTool(tool as AnyAgentTool, { optional: true });
  }

  api.registerCli(
    ({ program, config, logger, workspaceDir }) => {
      registerFinancialMarketCli(api, { program, config, logger, workspaceDir });
    },
    { commands: ["financial-market"] },
  );
}
