# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim

RUN useradd --create-home --shell /usr/sbin/nologin statmcp
COPY --from=builder /install /usr/local

ENV STATMCP_DATA_DIR=/data \
    STATMCP_PORT=8347 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data && chown statmcp:statmcp /data

USER statmcp
WORKDIR /data
EXPOSE 8347

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('STATMCP_PORT', '8347') + '/healthz', timeout=3)"]

ENTRYPOINT ["statistician-mcp", "--transport", "http"]
