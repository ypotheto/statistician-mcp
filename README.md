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
# {"status":"ok","version":"0.2.1"}
```

The MCP endpoint is served at `POST http://localhost:8347/mcp` (streamable HTTP,
protocol version `2025-03-26`).

### Testing locally without writing a client

- **Claude Desktop / Claude Code** — the stdio config block above; tools show up
  directly once the client restarts.
- **[MCP Inspector](https://github.com/modelcontextprotocol/inspector)** — the
  official dev tool, gives you a UI to call `tools/list`/`tools/call` without a
  full chat client: `npx @modelcontextprotocol/inspector .\.venv\Scripts\python.exe -m statistician_mcp --transport stdio`
  (or point it at the HTTP transport instead).
- **ChatGPT** — not covered here. ChatGPT is a cloud-hosted client and can't spawn
  a local stdio subprocess the way Claude Desktop can; it needs a reachable HTTPS
  URL (e.g. via ngrok/Cloudflare Tunnel) added as a custom connector, and exact
  connector requirements (OAuth, allow-listing) are outside what's verified here.

## Authentication

Two modes, set via `STATMCP_AUTH_MODE` (default `token`):

- **`token`** — a single static bearer token via `STATMCP_API_TOKEN`. Empty/unset
  disables auth entirely (dev only — this is what stdio/local testing above uses).
  Every valid request hashes to one shared workspace.
- **`keys`** — a real per-tenant API-key table (SQLite, at
  `{STATMCP_DATA_DIR}/keys.db`), each key resolving to its own workspace. Manage
  keys with the admin script:

  ```powershell
  .\.venv\Scripts\python.exe scripts\issue_key.py issue ws_acme --plan pro
  .\.venv\Scripts\python.exe scripts\issue_key.py list
  .\.venv\Scripts\python.exe scripts\issue_key.py disable sk_...
  ```

  The raw key is shown only once, at issuance — only its hash is stored.

`/healthz` is always public. `/artifacts/*` also accepts the token as a `?t=`
query parameter (browsers can't set an `Authorization` header on a plain link).

## Docker

```bash
docker build -t statistician-mcp .
docker run -p 8347:8347 -v statmcp-data:/data statistician-mcp
```

Runs as a non-root user, healthchecks `/healthz`. Set `STATMCP_AUTH_MODE=keys` and
mount `/data` persistently to keep the issued-key database across restarts.
