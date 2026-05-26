FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RADA_POLICY_DIR=/app/ml_server/config

WORKDIR /app

# ml_server is imported as a package (from .api... in main.py), so the
# WORKDIR must contain the `ml_server` directory and uvicorn is invoked
# as `ml_server.main:app` (matches systemd intent).
COPY ml_server/requirements.txt /app/ml_server/requirements.txt
RUN pip install -r /app/ml_server/requirements.txt

COPY ml_server /app/ml_server

EXPOSE 8000

CMD ["uvicorn", "ml_server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
