# Dockerfile for mnemostack.
#
# Builds a slim image with the mnemostack CLI on PATH. Use alongside Qdrant
# and Memgraph (see examples/docker-compose.yml), or on its own against
# external Qdrant/Memgraph/Ollama.
#
# Build:
#   docker build -t mnemostack:local .
#
# Run (one-shot, against local services on host):
#   docker run --rm --network host -e GEMINI_API_KEY=$GEMINI_API_KEY mnemostack:local \
#       health --provider gemini
#
# Run (interactive, with notes mounted):
#   docker run -it --rm --network host -v $(pwd)/notes:/data mnemostack:local \
#       index /data --provider gemini --collection demo

FROM python:3.12-slim

# Environment hardening + reproducible installs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System deps: curl for healthchecks, nothing else
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install from source (this is the dev / CI path — PyPI users can also
# just `pip install mnemostack` and skip the Dockerfile entirely).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# Run as a non-root user for anything that doesn't need root.
RUN useradd --create-home --shell /bin/bash mnemos
USER mnemos
WORKDIR /home/mnemos

# Default to showing the CLI help; operators override with a subcommand.
ENTRYPOINT ["mnemostack"]
CMD ["--help"]
