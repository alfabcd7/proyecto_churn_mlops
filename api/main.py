"""
main.py
API predictiva de churn con FastAPI + capa de observabilidad.

Endpoints:
  GET  /        → Información general del servicio
  GET  /health  → Estado de salud (modelo cargado, versión, features)
  GET  /info    → Metadata completa del modelo
  GET  /metrics → Métricas operativas en tiempo real (observabilidad)
  POST /predict → Predicción de churn con nivel de riesgo y recomendación
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from collections import defaultdict

import joblib
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

# ══════════════════════════════════════════
# Configuración
# ══════════════════════════════════════════
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_FILE = os.path.join(MODELS_DIR, "modelo_churn_v1.joblib")
METADATA_FILE = os.path.join(MODELS_DIR, "modelo_churn_v1_metadata.json")

AUTOR = "Carlos TuApellido"  # ← CAMBIAR por tu nombre y apellido real
VERSION_SERVICIO = "1.0.0"

# ══════════════════════════════════════════
# Logging estructurado
# ══════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("churn-api")

# ══════════════════════════════════════════
# Métricas en memoria (observabilidad)
# ══════════════════════════════════════════
metricas_servicio = {
    "inicio_servicio": datetime.now(timezone.utc).isoformat(),
    "total_predicciones": 0,
    "total_errores_validacion": 0,
    "predicciones_por_nivel": defaultdict(int),
    "suma_probabilidades": 0.0,
    "latencias_ms": [],
    "ultima_prediccion": None,
}

# ══════════════════════════════════════════
# Cargar modelo y metadata
# ══════════════════════════════════════════
logger.info("Cargando modelo desde %s", MODEL_FILE)
modelo = joblib.load(MODEL_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

logger.info(
    "Modelo cargado: version=%s, features=%s",
    metadata.get("version"),
    metadata.get("features"),
)

# ══════════════════════════════════════════
# App FastAPI
# ══════════════════════════════════════════
app = FastAPI(
    title="API Predictiva de Churn — ML-Ops",
    description="Servicio de predicción de abandono de clientes con observabilidad.",
    version=VERSION_SERVICIO,
)


# ══════════════════════════════════════════
# Middleware: registrar errores de validación
# ══════════════════════════════════════════
@app.middleware("http")
async def registrar_errores(request: Request, call_next):
    response = await call_next(request)
    if response.status_code == 422:
        metricas_servicio["total_errores_validacion"] += 1
        logger.warning(
            "Solicitud rechazada | endpoint=%s | status=422",
            request.url.path,
        )
    return response


# ══════════════════════════════════════════
# Esquema de entrada
# ══════════════════════════════════════════
class ClienteInput(BaseModel):
    """Datos del cliente para predicción de churn."""
    edad: int = Field(
        ..., ge=18, le=100,
        description="Edad del cliente (18-100)",
        json_schema_extra={"example": 35},
    )
    antiguedad_meses: int = Field(
        ..., ge=0, le=120,
        description="Meses como cliente (0-120)",
        json_schema_extra={"example": 24},
    )
    saldo_promedio: float = Field(
        ..., ge=0, le=50000,
        description="Saldo promedio en cuenta (0-50000)",
        json_schema_extra={"example": 2500.0},
    )
    reclamos: int = Field(
        ..., ge=0, le=50,
        description="Cantidad de reclamos (0-50)",
        json_schema_extra={"example": 1},
    )
    usa_app: int = Field(
        ..., ge=0, le=1,
        description="1 si usa la app móvil, 0 si no",
        json_schema_extra={"example": 1},
    )


# ══════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════

@app.get("/")
def inicio():
    """Estado general del servicio."""
    return {
        "mensaje": "Servicio ML-Ops activo",
        "estado": "ok",
        "autor": AUTOR,
        "version": VERSION_SERVICIO,
    }


@app.get("/health")
def health():
    """Verificación de salud: confirma que el modelo está cargado y listo."""
    return {
        "status": "healthy",
        "modelo_cargado": modelo is not None,
        "version_modelo": metadata.get("version", "desconocida"),
        "features_esperadas": metadata.get("features", []),
        "autor": AUTOR,
    }


@app.get("/info")
def info():
    """Metadata completa del modelo entrenado (mejora técnica 1)."""
    return {
        "autor": AUTOR,
        "version_servicio": VERSION_SERVICIO,
        "modelo": metadata.get("modelo"),
        "version_modelo": metadata.get("version"),
        "fecha_entrenamiento": metadata.get("fecha_entrenamiento"),
        "hiperparametros": metadata.get("hiperparametros"),
        "features": metadata.get("features"),
        "metricas": metadata.get("metricas"),
        "datos": metadata.get("datos"),
    }


@app.get("/metrics")
def metrics():
    """
    Métricas operativas en tiempo real (observabilidad).

    Expone contadores de predicciones, distribución por nivel de riesgo,
    probabilidad promedio, latencia promedio y errores de validación.
    Permite monitorear la salud operativa del servicio y detectar drift.
    """
    total = metricas_servicio["total_predicciones"]
    latencias = metricas_servicio["latencias_ms"]

    prob_promedio = (
        round(metricas_servicio["suma_probabilidades"] / total, 4)
        if total > 0 else None
    )

    latencia_promedio_ms = (
        round(sum(latencias) / len(latencias), 2)
        if latencias else None
    )

    latencia_p95_ms = (
        round(sorted(latencias)[int(len(latencias) * 0.95)], 2)
        if len(latencias) >= 20 else None
    )

    return {
        "servicio": {
            "inicio": metricas_servicio["inicio_servicio"],
            "version": VERSION_SERVICIO,
            "autor": AUTOR,
        },
        "predicciones": {
            "total": total,
            "por_nivel": dict(metricas_servicio["predicciones_por_nivel"]),
            "probabilidad_promedio_churn": prob_promedio,
            "ultima_prediccion": metricas_servicio["ultima_prediccion"],
        },
        "rendimiento": {
            "latencia_promedio_ms": latencia_promedio_ms,
            "latencia_p95_ms": latencia_p95_ms,
        },
        "errores": {
            "validacion_422": metricas_servicio["total_errores_validacion"],
        },
    }


@app.post("/predict")
def predecir(cliente: ClienteInput):
    """
    Predicción de churn con nivel de riesgo, recomendación
    y registro de métricas (observabilidad).
    """
    inicio = time.perf_counter()

    datos = np.array([[
        cliente.edad,
        cliente.antiguedad_meses,
        cliente.saldo_promedio,
        cliente.reclamos,
        cliente.usa_app,
    ]])

    probabilidad = float(modelo.predict_proba(datos)[0][1])

    if probabilidad >= 0.75:
        nivel = "critico"
        recomendacion = "Contactar al cliente de forma inmediata con oferta de retención personalizada."
    elif probabilidad >= 0.50:
        nivel = "alto"
        recomendacion = "Priorizar seguimiento y evaluar beneficios adicionales."
    elif probabilidad >= 0.25:
        nivel = "moderado"
        recomendacion = "Monitorear comportamiento en los próximos 30 días."
    else:
        nivel = "bajo"
        recomendacion = "Cliente estable. Mantener calidad de servicio habitual."

    # ── Registrar métricas ──
    latencia_ms = (time.perf_counter() - inicio) * 1000
    metricas_servicio["total_predicciones"] += 1
    metricas_servicio["predicciones_por_nivel"][nivel] += 1
    metricas_servicio["suma_probabilidades"] += probabilidad
    metricas_servicio["latencias_ms"].append(latencia_ms)
    metricas_servicio["ultima_prediccion"] = datetime.now(timezone.utc).isoformat()

    # Mantener solo las últimas 1000 latencias para no consumir memoria
    if len(metricas_servicio["latencias_ms"]) > 1000:
        metricas_servicio["latencias_ms"] = metricas_servicio["latencias_ms"][-500:]

    # ── Log estructurado ──
    logger.info(
        "Prediccion | prob=%.4f | nivel=%s | edad=%d | antiguedad=%d | reclamos=%d | usa_app=%d | latencia=%.2fms",
        probabilidad, nivel,
        cliente.edad, cliente.antiguedad_meses,
        cliente.reclamos, cliente.usa_app,
        latencia_ms,
    )

    return {
        "prediccion": "churn" if probabilidad >= 0.5 else "no_churn",
        "probabilidad_churn": round(probabilidad, 4),
        "nivel_riesgo": nivel,
        "recomendacion": recomendacion,
        "datos_recibidos": cliente.model_dump(),
    }