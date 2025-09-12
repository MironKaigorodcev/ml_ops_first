import json
import logging
import os
import time
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, ConfigDict
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from starlette.concurrency import run_in_threadpool

# -----------------------------
# Конфигурация
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/model_gb.pkl")
SERVICE_NAME = os.getenv("SERVICE_NAME", "ml-inference-service")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# -----------------------------
# Логирование (JSON)
# -----------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "level": record.levelname,
            "time": int(time.time()),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record, ensure_ascii=False)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger(SERVICE_NAME)
logger.setLevel(LOG_LEVEL)
logger.handlers = [handler]
logger.propagate = False

# -----------------------------
# Метрики Prometheus
# -----------------------------
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Latency of HTTP requests in seconds",
    labelnames=("method", "endpoint", "status"),
)
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total number of prediction calls",
    labelnames=("status",),
)

# -----------------------------
# Модель: лениво загружаем и кэшируем
# -----------------------------
_model = None
_model_loaded_at: Optional[float] = None

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

class HeartFeatures(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: int
    request_id: Optional[str] = Field(None, description="Client request id for tracing")

class PredictResponse(BaseModel):
    # отключаем защищённый префикс "model_" в Pydantic v2 (иначе warning)
    model_config = ConfigDict(protected_namespaces=())
    prediction: float | int | str
    proba: Optional[float] = None
    model_path: str
    model_loaded_at: Optional[float]
    request_id: Optional[str]

def load_model():
    global _model, _model_loaded_at
    if _model is not None:
        return _model
    logger.info(f"Загружаю модель из {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        logger.exception("Ошибка загрузки модели")
        raise RuntimeError(f"Failed to load model: {e}")
    _model = model
    _model_loaded_at = time.time()
    return _model

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title=SERVICE_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # при необходимости ограничь домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# redirect корня на /docs
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

# метрики латентности
@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.time() - start
        endpoint = request.url.path
        REQUEST_LATENCY.labels(request.method, endpoint, str(status)).observe(elapsed)

# служебные эндпоинты
@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": SERVICE_NAME}

@app.get("/readyz")
async def readyz():
    try:
        model = await run_in_threadpool(load_model)
        if not hasattr(model, "predict"):
            raise RuntimeError("Model has no predict method")
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"not ready: {e}")

@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

# инференс
@app.post("/predict", response_model=PredictResponse)
async def predict(payload: HeartFeatures):
    try:
        model = await run_in_threadpool(load_model)
        row = [getattr(payload, f) for f in FEATURES]
        X = np.array([row], dtype=float)

        def _do_predict():
            y = model.predict(X)
            proba_local = None
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)
                if p.ndim == 2 and p.shape[1] > 1:
                    proba_local = float(p[0, 1])  # вероятность положительного класса
                else:
                    proba_local = float(p[0])
            return y[0], proba_local

        pred, proba = await run_in_threadpool(_do_predict)
        PREDICTIONS_TOTAL.labels("success").inc()

        # лог полезной инфы
        logger.info(
            f"predict ok; request_id={payload.request_id}; pred={pred}; proba={proba}"
        )

        # аккуратно приводим тип numpy -> питоновский
        pred_value = pred.item() if hasattr(pred, "item") else pred

        return PredictResponse(
            prediction=pred_value,
            proba=proba,
            model_path=MODEL_PATH,
            model_loaded_at=_model_loaded_at,
            request_id=payload.request_id,
        )
    except Exception as e:
        logger.exception("Ошибка во время предикта")
        PREDICTIONS_TOTAL.labels("error").inc()
        raise HTTPException(status_code=500, detail=str(e))

# Локальный запуск:
# uvicorn app:app --host 0.0.0.0 --port 8000
