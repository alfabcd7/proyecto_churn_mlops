"""
test_api.py
Pruebas del servicio predictivo de churn con observabilidad.
Cubre: endpoints básicos, predicciones, validaciones y métricas.

Ejecutar:  pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ═══════════════════════════════════════
# GET /
# ═══════════════════════════════════════
class TestRaiz:
    def test_raiz_responde_ok(self):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["estado"] == "ok"
        assert "autor" in body


# ═══════════════════════════════════════
# GET /health
# ═══════════════════════════════════════
class TestHealth:
    def test_health_modelo_cargado(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["modelo_cargado"] is True
        assert isinstance(body["features_esperadas"], list)


# ═══════════════════════════════════════
# GET /info (mejora personal)
# ═══════════════════════════════════════
class TestInfo:
    def test_info_contiene_metadata(self):
        resp = client.get("/info")
        assert resp.status_code == 200
        body = resp.json()
        assert "version_modelo" in body
        assert "fecha_entrenamiento" in body
        assert "metricas" in body
        assert "hiperparametros" in body


# ═══════════════════════════════════════
# GET /metrics (observabilidad)
# ═══════════════════════════════════════
class TestMetrics:
    def test_metrics_estructura(self):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "servicio" in body
        assert "predicciones" in body
        assert "rendimiento" in body
        assert "errores" in body

    def test_metrics_conteo_despues_de_prediccion(self):
        """Verificar que /metrics incrementa el contador tras una predicción."""
        # Obtener conteo actual
        antes = client.get("/metrics").json()["predicciones"]["total"]

        # Hacer una predicción
        payload = {"edad": 30, "antiguedad_meses": 12, "saldo_promedio": 2000, "reclamos": 1, "usa_app": 1}
        client.post("/predict", json=payload)

        # Verificar incremento
        despues = client.get("/metrics").json()["predicciones"]["total"]
        assert despues == antes + 1

    def test_metrics_registra_errores_422(self):
        """Verificar que /metrics cuenta los errores de validación."""
        antes = client.get("/metrics").json()["errores"]["validacion_422"]

        # Provocar un error 422
        client.post("/predict", json={"edad": "texto_invalido"})

        despues = client.get("/metrics").json()["errores"]["validacion_422"]
        assert despues >= antes + 1


# ═══════════════════════════════════════
# POST /predict — solicitud válida
# ═══════════════════════════════════════
class TestPrediccionValida:
    def test_prediccion_cliente_valido(self):
        payload = {
            "edad": 28,
            "antiguedad_meses": 8,
            "saldo_promedio": 1200.0,
            "reclamos": 3,
            "usa_app": 0,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "prediccion" in body
        assert "probabilidad_churn" in body
        assert "nivel_riesgo" in body
        assert "recomendacion" in body
        assert body["prediccion"] in ("churn", "no_churn")
        assert 0.0 <= body["probabilidad_churn"] <= 1.0

    def test_prediccion_cliente_bajo_riesgo(self):
        payload = {
            "edad": 45,
            "antiguedad_meses": 60,
            "saldo_promedio": 4500.0,
            "reclamos": 0,
            "usa_app": 1,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["nivel_riesgo"] in ("bajo", "moderado", "alto", "critico")


# ═══════════════════════════════════════
# POST /predict — campo faltante
# ═══════════════════════════════════════
class TestCampoFaltante:
    def test_sin_edad(self):
        payload = {
            "antiguedad_meses": 12,
            "saldo_promedio": 2000.0,
            "reclamos": 1,
            "usa_app": 1,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422
        campos_error = [e["loc"][-1] for e in resp.json()["detail"]]
        assert "edad" in campos_error

    def test_sin_reclamos(self):
        payload = {
            "edad": 30,
            "antiguedad_meses": 12,
            "saldo_promedio": 2000.0,
            "usa_app": 1,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


# ═══════════════════════════════════════
# POST /predict — tipo de dato incorrecto
# ═══════════════════════════════════════
class TestTipoIncorrecto:
    def test_edad_como_texto(self):
        payload = {
            "edad": "veintiocho",
            "antiguedad_meses": 8,
            "saldo_promedio": 1200.0,
            "reclamos": 3,
            "usa_app": 0,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_saldo_como_texto(self):
        payload = {
            "edad": 28,
            "antiguedad_meses": 8,
            "saldo_promedio": "mil doscientos",
            "reclamos": 3,
            "usa_app": 0,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


# ═══════════════════════════════════════
# POST /predict — valor fuera de rango
# ═══════════════════════════════════════
class TestFueraDeRango:
    def test_edad_menor_a_18(self):
        payload = {"edad": 10, "antiguedad_meses": 12, "saldo_promedio": 2000, "reclamos": 1, "usa_app": 1}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_reclamos_negativo(self):
        payload = {"edad": 30, "antiguedad_meses": 12, "saldo_promedio": 2000, "reclamos": -3, "usa_app": 1}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_usa_app_fuera_de_rango(self):
        payload = {"edad": 30, "antiguedad_meses": 12, "saldo_promedio": 2000, "reclamos": 1, "usa_app": 5}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_edad_mayor_a_100(self):
        payload = {"edad": 150, "antiguedad_meses": 12, "saldo_promedio": 2000, "reclamos": 1, "usa_app": 1}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422