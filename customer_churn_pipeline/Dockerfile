# Multi-stage production Dockerfile for Customer Churn API
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

# Create unprivileged user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application and artifacts
COPY src/ /app/src/
COPY app/ /app/app/
COPY artifacts/ /app/artifacts/
COPY data/ /app/data/
COPY requirements.txt /app/

# Environment configuration
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Fix permissions
RUN chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start FastAPI production server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
