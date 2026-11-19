# Geospatial stack needs system GDAL/GEOS/PROJ; the slim image plus wheels is enough.
FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir \
      "geopandas>=1.0" "shapely>=2.0" "pyogrio>=0.8" "pyproj>=3.6" \
      "networkx>=3.0" "numpy>=1.26" "pandas>=2.0" "pyarrow>=15.0" \
      "matplotlib>=3.8" "requests>=2.31" "fastapi>=0.110" "uvicorn[standard]>=0.29"

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY api/ ./api/
COPY tests/ ./tests/

# Licensed OS data is NOT baked into the image. Mount it:
#   docker run -v $PWD/data:/app/data -v $PWD/outputs:/app/outputs ...
ENV SIDEWALK_DATA_DIR=/app/data SIDEWALK_OUTPUT_DIR=/app/outputs
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
