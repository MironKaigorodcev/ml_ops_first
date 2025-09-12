# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1 \
PIP_NO_CACHE_DIR=1 \
PIP_DISABLE_PIP_VERSION_CHECK=1

# Создадим непривилегированного пользователя
ARG APP_UID=10001
RUN useradd -m -u ${APP_UID} -s /usr/sbin/nologin appuser

WORKDIR /app

# Устанавливаем системные зависимости по минимуму
RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
&& rm -rf /var/lib/apt/lists/*

# Кэшируем слои зависимостей
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Копируем код
COPY app.py ./
# встраиваем модель в образ
COPY models/ ./models/

# Безопасные права
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# HEALTHCHECK  на чистом Python
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import sys,urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3); sys.exit(0 if r.getcode()==200 else 1)" || exit 1

# Переменные окружения настраиваемы при docker run / compose
ENV MODEL_PATH=/app/models/model_gb.pkl \
SERVICE_NAME=ml-inference-service \
LOG_LEVEL=INFO \
WORKERS=2

# Запуск через gunicorn с uvicorn worker'ами
CMD gunicorn \
--workers "${WORKERS}" \
--worker-class uvicorn.workers.UvicornWorker \
--bind 0.0.0.0:8000 \
--timeout 60 \
app:app