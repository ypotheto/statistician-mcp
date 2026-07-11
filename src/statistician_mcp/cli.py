from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import uvicorn

from statistician_mcp.config import get_settings
from statistician_mcp.http_app import create_app
from statistician_mcp.server import create_server


def build_arg_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="statistician-mcp")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="MCP transport to serve: stdio (local clients) or http (streamable HTTP).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help="Port to bind for the http transport (default: %(default)s).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.data_dir,
        help="Directory for datasets, artifacts, and other persistent state.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    settings = get_settings()
    settings.port = args.port
    settings.data_dir = args.data_dir
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    mcp = create_server(settings)

    if args.transport == "stdio":
        asyncio.run(mcp.run_stdio_async())
    else:
        app = create_app(mcp)
        uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
