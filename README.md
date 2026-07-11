# statistician-mcp
An MCP server providing virtual statistician tools for DOE, SPC, EDA, statistical modeling, and data-driven decision support.

See [planning/statistician_mcp_plan.md](planning/statistician_mcp_plan.md) for the full
architecture and phase-by-phase implementation plan.

## Quickstart

Install in editable mode with dev dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

### stdio (Claude Desktop / Claude Code)

Add to your MCP client config (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "statistician": {
      "command": "C:\\path\\to\\statistician-mcp\\.venv\\Scripts\\python.exe",
      "args": ["-m", "statistician_mcp", "--transport", "stdio"]
    }
  }
}
```

### Streamable HTTP (ypotheto-core, hosted deployments)

```powershell
.\.venv\Scripts\python.exe -m statistician_mcp --transport http --port 8347
```

```bash
curl http://localhost:8347/healthz
# {"status":"ok","version":"0.1.0"}
```

The MCP endpoint is served at `POST http://localhost:8347/mcp` (streamable HTTP,
protocol version `2025-03-26`). Authentication (`STATMCP_API_TOKEN`) and the artifact
store are added in Phase 1 — see the plan doc for details.
