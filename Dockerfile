FROM python:3.11-slim  
  
ENV PYTHONUNBUFFERED=1 \  
    PYTHONDONTWRITEBYTECODE=1  
  
RUN apt-get update && apt-get install -y curl \  
    && rm -rf /var/lib/apt/lists/*  
  
WORKDIR /app  
  
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  
  
COPY backend/ ./backend/  
COPY config.yaml .  
  
# Create runtime dirs and set ownership before switching user  
RUN useradd --create-home --shell /bin/bash appuser \  
    && mkdir -p data/pdfs data/faiss_index .cache/embeddings .cache/responses logs \  
    && chown -R appuser:appuser /app  
  
USER appuser  
  
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \  
    CMD curl -f http://localhost:8000/health || exit 1  
  
EXPOSE 8000  
  
CMD ["uvicorn", "backend.main:app", \  
     "--host", "0.0.0.0", \  
     "--port", "8000", \  
     "--workers", "1", \  
     "--access-log"]
     