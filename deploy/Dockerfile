FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
    && pip install --no-cache-dir -r deploy/requirements.txt
COPY src /app/src
COPY models/current_models /app/models/current_models
COPY tools/export_compact_forest.py /app/tools/export_compact_forest.py
# Build derived arrays from the exact checkpoint. Runtime never loads the
# training-only forest object. The original model remains in the image.
RUN python tools/export_compact_forest.py models/current_models/random_forest.joblib /app/runtime_forest
ENV BRFSS_COMPACT_FOREST_DIR=/app/runtime_forest
COPY reports/brfss2022_expanded/dataset_audit.json reports/brfss2022_expanded/deployment_gate.json /app/reports/brfss2022_expanded/
COPY deploy/backend.py deploy/__init__.py /app/deploy/
RUN useradd --create-home appuser
USER appuser
# Keep the master lightweight: only the single worker owns the inference stack.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 120 deploy.backend:application"]
