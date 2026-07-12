# statistician-mcp

**A virtual statistician for your AI agent.** An [MCP](https://modelcontextprotocol.io)
server that gives Claude, ChatGPT, or any MCP-speaking client real statistical
methods to work with — design of experiments, statistical process control,
hypothesis testing, regression, measurement systems analysis — instead of
hoping the model remembers the right formula and doesn't hallucinate a p-value.

MIT licensed. Built on `scipy`, `statsmodels`, and `pyDOE3` — the same
libraries a human statistician would reach for, just wired up so an LLM can
call them directly.

## Why this is different from "ask the model to do stats"

- **Real methods, not guesses.** Every tool runs an actual, tested statistical
  routine (Shapiro-Wilk, Tukey HSD, Gauge R&R variance components, D-optimal
  and factorial designs, Western Electric/Nelson rules for control charts) —
  the kind of thing that's easy for an LLM to get subtly wrong if it's doing
  the arithmetic itself, and easy to get right when it's calling a library
  built for it.
- **Every result explains itself.** Tool responses don't just return numbers —
  they come back with a plain-language `interpretation` field ("this p-value
  means...", "Cpk of 1.1 means..."), so the agent (and the human reading its
  output) doesn't have to guess what a statistic means.
- **A built-in advisor, not just a toolbox.** `recommend_analysis` looks at
  what you're trying to answer and suggests which test or design actually
  fits; `explain_concept` gives a plain-English explanation of ~30 core
  statistical concepts on demand.
- **Broader than a stats library wrapper.** Most "do some stats" integrations
  stop at t-tests and correlation. This covers full design-of-experiments
  (factorial, response surface, optimization), SPC (control charts,
  capability, stability rules), and MSA (Gauge R&R) — the parts of applied
  statistics that manufacturing, quality, and experimental-science work
  actually needs, and that a general-purpose model rarely gets right
  unprompted.
- **Plots, not just numbers.** Distribution plots, control charts, response
  surfaces, and power curves render as real images the agent can hand back to
  you, not ASCII art.
- **Works everywhere MCP works.** stdio for Claude Desktop/Claude Code, or
  streamable HTTP for a hosted deployment any MCP client can point at.

## What's in the box

| Category | Tools |
|---|---|
| **Datasets** | load from CSV/URL, list, describe, sample, transform, delete |
| **EDA** | column summaries, distribution plots, normality tests, outlier detection, correlations, scatter/time-series plots, crosstabs |
| **Inference & power** | compare means/proportions/variances, compare multiple groups, equivalence testing, confidence intervals, power/sample-size calculations |
| **Design of experiments** | design factorial/response-surface experiments, evaluate a design, analyze factorial results, analyze response surfaces, optimize a response |
| **Statistical process control** | control charts (X-bar/R, I-MR, ...), process capability (Cp/Cpk/Pp/Ppk), stability/out-of-control rule checks |
| **Measurement systems analysis** | Gauge R&R (crossed ANOVA), attribute agreement analysis |
| **Regression & modeling** | linear/logistic models, model comparison, prediction, distribution fitting |
| **Advisor** | recommend which analysis fits your question, explain a statistical concept |

39 tools in total. See the module source under
[`src/statistician_mcp/modules/`](src/statistician_mcp/modules/) for the full
list and docstrings.

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

### Streamable HTTP (hosted deployments)

```powershell
.\.venv\Scripts\python.exe -m statistician_mcp --transport http --port 8347
```

```bash
curl http://localhost:8347/healthz
# {"status":"ok","version":"0.2.2"}
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
- **ChatGPT** — connects over the HTTP transport as a custom connector; needs a
  reachable HTTPS URL and a bearer token (`keys` mode below).

## Authentication

Three modes, set via `STATMCP_AUTH_MODE` (default `token`):

- **`token`** — a single static bearer token via `STATMCP_API_TOKEN`. Empty/unset
  disables auth entirely (dev only — this is what stdio/local testing above uses).
  Every valid request hashes to one shared workspace.
- **`keys`** — a real per-tenant API-key table, each key resolving to its own
  workspace. SQLite at `{STATMCP_DATA_DIR}/keys.db` by default; set
  `STATMCP_DATABASE_URL` to a Postgres DSN to use `PostgresKeyStore` instead (a
  hosted deployment's natural choice — the table is created automatically on
  first connect, in whatever schema the DSN's role defaults to via its
  `search_path`). Manage keys with the admin script (targets whichever store the
  server would use; `--db PATH` overrides to force SQLite at that path):

  ```powershell
  .\.venv\Scripts\python.exe scripts\issue_key.py issue ws_acme --plan pro
  .\.venv\Scripts\python.exe scripts\issue_key.py list
  .\.venv\Scripts\python.exe scripts\issue_key.py disable sk_...
  ```

  The raw key is shown only once, at issuance — only its hash is stored.

- **`oauth`** — for clients (like Claude's custom connectors) that speak OAuth
  rather than a static bearer token. This server only ever plays the OAuth
  *resource server* role — an external identity provider (Kinde, or any OIDC
  provider) is the authorization server; there's no `/authorize` or `/token`
  endpoint here, just JWT validation. Requires:

  ```
  STATMCP_OAUTH_ISSUER=https://<your-tenant>.kinde.com
  STATMCP_OAUTH_AUDIENCE=https://<this-server>/mcp
  STATMCP_OAUTH_REQUIRED_PERMISSION=access:statistician-mcp   # optional, this is already the default
  ```

  A token must be signed by the issuer, carry the exact `aud` above, and
  include `STATMCP_OAUTH_REQUIRED_PERMISSION` in its `permissions` claim —
  this last check is the actual access gate (assigned per-user in the
  provider's dashboard today; swappable later for automatic
  subscription-driven entitlements without any code change here). Each
  distinct authenticated user gets their own workspace, derived from the
  token's `sub` claim the same way the `token` mode derives one from the
  static token. `GET /.well-known/oauth-protected-resource` (RFC 9728) tells
  a client where to authenticate; a 401 in this mode carries a
  `WWW-Authenticate` header pointing at it.

`/healthz` and `/.well-known/oauth-protected-resource` are always public.
`/artifacts/*` also accepts the token as a `?t=` query parameter (browsers
can't set an `Authorization` header on a plain link).

### Connecting from Claude (`oauth` mode)

Claude Desktop / claude.ai support custom MCP connectors with OAuth:

1. **Settings → Connectors → Add custom connector**
2. **Remote MCP server URL**: your deployment's MCP endpoint, e.g.
   `https://your-deployment.example.com/mcp`
3. **Advanced settings → OAuth Client ID / OAuth Client Secret**: from
   whatever OAuth application you registered in your identity provider (a
   "back-end web" / "regular web app" type, *not* machine-to-machine — this
   flow needs a real user login/consent screen and an individual identity,
   which M2M apps don't have) for this MCP server, with the redirect URI set
   to `https://claude.ai/api/mcp/auth_callback`
4. Click **Add**, then enable the connector for a conversation via the **+**
   button, and Claude will walk you through logging in.

This only works if the server operator has set `STATMCP_AUTH_MODE=oauth` and
configured an identity provider (see above) — the `token`/`keys` modes don't
speak OAuth at all, and need a bearer token supplied directly instead (via a
connector's plain API-key/header auth option, where the client supports it).

Other MCP clients (ChatGPT, etc.) that support custom connectors follow a
similar shape — point them at the same `/mcp` URL and either the OAuth flow
above or a bearer token, depending on what the server is configured for and
what the client supports. This section will grow as more of them get verified
against this server.

## Docker

```bash
docker build -t statistician-mcp .
docker run -p 8347:8347 -v statmcp-data:/data statistician-mcp
```

Runs as a non-root user, healthchecks `/healthz`. Set `STATMCP_AUTH_MODE=keys` and
mount `/data` persistently to keep the issued-key database across restarts.

## Storage backend

Dataset/artifact storage defaults to local disk under `{STATMCP_DATA_DIR}/storage`
(`LocalDirBackend`) — fine for a Droplet with a mounted volume, but incompatible
with ephemeral-disk compute like DO App Platform. Setting `STATMCP_SPACES_BUCKET`
switches to a DigitalOcean Spaces bucket (`SpacesBackend`) instead; all of
endpoint/key/secret must be set together:

```
STATMCP_SPACES_BUCKET=my-bucket
STATMCP_SPACES_ENDPOINT=https://nyc3.digitaloceanspaces.com
STATMCP_SPACES_KEY=...
STATMCP_SPACES_SECRET=...
STATMCP_SPACES_REGION=nyc3          # optional, defaults to nyc3
STATMCP_SPACES_PREFIX=statistician-mcp   # optional, this is already the default
```

If the bucket is shared with other services (each their own MCP server, say),
`STATMCP_SPACES_PREFIX` namespaces every key this app writes under e.g.
`statistician-mcp/...`, so two services can point at the same bucket with
different prefixes and never see or collide with each other's objects — give
each service its own prefix (and, on the DO side, its own Spaces access key
scoped to just that bucket via `doctl spaces keys create <name> --grants
'bucket=my-bucket;permission=readwrite'`).

`STATMCP_DATABASE_URL` (see Authentication above) is independent of this
setting — one config switch for the key table, one for dataset/artifact
storage; a deployment can mix and match (e.g. Postgres keys + local-disk
storage on a Droplet, or Postgres keys + Spaces storage for App Platform).

## Deployment

Two viable paths on DigitalOcean — pick based on how much ops you want:

- **App Platform** (this repo's spec: [`.do/app.yaml`](.do/app.yaml)) — deploys
  the Dockerfile directly from this GitHub repo, managed TLS, no server to
  patch. Requires `STATMCP_DATABASE_URL` and the `STATMCP_SPACES_*` vars set
  (App Platform's disk is ephemeral, so local SQLite/local-dir storage don't
  survive a redeploy). Set each `STATMCP_*` secret as an encrypted app-level
  env var — never commit real values into `.do/app.yaml`, which only holds
  placeholders. Pick a region that co-locates with wherever your Postgres
  cluster and Spaces bucket actually live, to keep the per-request key lookup
  and dataset/artifact I/O both low-latency.
- **Droplet + Docker** — a small droplet with a mounted volume keeps the
  local-dir storage backend and SQLite key store working unchanged, at the
  cost of your own reverse proxy (Caddy/nginx + Let's Encrypt) for TLS and OS
  upkeep. Fine for a private beta; App Platform is the better path once
  you need to scale beyond one box.

Either way: set `STATMCP_PUBLIC_BASE_URL` to the real public HTTPS URL once
it's known (artifact links resolve against this — App Platform has no
bindable variable for an app's own URL, so this has to be set as a second step
after the first deploy, once DO tells you the assigned domain) and smoke-test
`/healthz` plus an actual tool call against the hosted instance before
declaring it done.

See [CHANGELOG.md](CHANGELOG.md) for what's been built and verified so far.

## Contributing

Issues and pull requests welcome. Run `ruff check .`, `mypy src`, and `pytest`
before submitting — CI runs the same three.

## License

[MIT](LICENSE)
