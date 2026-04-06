FROM python:3.10-slim

# Metadata
LABEL maintainer="dataclean-openenv"
LABEL version="1.0.0"
LABEL description="DataClean OpenEnv — Real-world data cleaning environment for AI agents"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create __init__ files
#RUN touch environment/__init__.py graders/__init__.py
# Create folders and then create the __init__ files
RUN mkdir -p environment graders && touch environment/__init__.py graders/__init__.py
# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
