FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

RUN python -m pip install --upgrade pip && \
    python -m pip install .

ENTRYPOINT ["sdlc-agent"]
