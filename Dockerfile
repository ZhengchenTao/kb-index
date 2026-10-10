# Indexer image. The configuration is not baked in: mount it at /config and the data directory at /data.
#
#   docker run --rm \
#     -v ./sources.yaml:/config/sources.yaml:ro -v ./.indexignore:/config/.indexignore:ro \
#     -v /path/to/repos:/sources:ro -v kb-data:/data \
#     -e GEMINI_API_KEY kb-index index
#
# /data holds work.db (with the vector cache), the published index.db + index.db.sha256 + deltas/ +
# index.manifest.json, and embed-usage.jsonl.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY kbindex ./kbindex
# Slow or flaky links to the package index: a read timeout during resolution can surface as a
# misleading "ResolutionImpossible". Longer timeout and more retries make that go away.
ENV PIP_DEFAULT_TIMEOUT=60 PIP_RETRIES=10
RUN pip install --no-cache-dir .

ENV KB_CONFIG=/config/sources.yaml \
    KB_DB=/data/work.db \
    KB_OUT=/data \
    PYTHONUNBUFFERED=1

# Repositories are mounted read-only and may be owned by another uid; git needs safe.directory,
# which the code passes on every call (-c safe.directory=*).
ENTRYPOINT ["idx"]
CMD ["index"]
