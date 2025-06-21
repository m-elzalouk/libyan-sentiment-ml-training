# Dockerfile for Libyan Dialect Sentiment Analysis
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy scripts and data
COPY scripts/ scripts/
COPY data/ data/

# Default command: run training script
CMD [".venv/bin/python", "-m", "scripts.train"]
