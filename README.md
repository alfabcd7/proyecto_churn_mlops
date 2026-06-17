# proyecto-churn-mlops

Servicio de predicción de abandono de clientes (churn) construido con FastAPI, scikit-learn y Docker como parte del Módulo 14 — ML-Ops y Puesta en Producción.

## Estructura del proyecto

```
proyecto-churn-mlops-tuapellido/
├── api/
│   └── main.py              # API FastAPI: /, /health, /info, /predict
├── src/
│   ├── preparar_datos.py     # Genera dataset sintético y split train/test
│   └── entrenar_modelo.py    # Entrena RandomForest, serializa y genera metadata
├── tests/
│   └── test_api.py           # Pruebas: solicitudes válidas e inválidas
├── models/                   # Modelo .joblib y metadata .json (generados)
├── data/                     # Datasets CSV (generados)
├── docs/                     # Reporte de métricas (generado)
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Ejecución local

```bash
# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Preparar datos y entrenar modelo
python src/preparar_datos.py
python src/entrenar_modelo.py

# Iniciar la API
python -m uvicorn api.main:app --reload
```

Abrir en el navegador: http://127.0.0.1:8000/docs

## Ejecución con Docker

```bash
docker build -t churn-api-tuapellido:0.1 .
docker run --name churn-api -p 8000:8000 churn-api-tuapellido:0.1
```

## Endpoints

| Método | Ruta      | Descripción                                      |
|--------|-----------|--------------------------------------------------|
| GET    | /         | Estado general del servicio y autor               |
| GET    | /health   | Verificación de salud: modelo cargado y features  |
| GET    | /info     | Metadata completa del modelo (mejora personal)    |
| POST   | /predict  | Predicción de churn con nivel de riesgo           |

### Ejemplo POST /predict

**Solicitud:**
```json
{
  "edad": 28,
  "antiguedad_meses": 8,
  "saldo_promedio": 1200.0,
  "reclamos": 3,
  "usa_app": 0
}
```

**Respuesta:**
```json
{
  "prediccion": "churn",
  "probabilidad_churn": 0.6823,
  "nivel_riesgo": "alto",
  "recomendacion": "Priorizar seguimiento y evaluar beneficios adicionales.",
  "datos_recibidos": { ... }
}
```

### Validaciones implementadas

La API rechaza solicitudes con:
- Campos faltantes (HTTP 422)
- Tipos de dato incorrectos, por ejemplo edad como texto (HTTP 422)
- Valores fuera de rango, por ejemplo edad < 18 o reclamos negativos (HTTP 422)

## Pruebas automatizadas

```bash
pytest tests/test_api.py -v
```

Las pruebas cubren: solicitud válida, campo faltante, tipo incorrecto y valor fuera de rango.

## Mejora técnica personal

Se implementaron dos mejoras:

1. **Endpoint /info**: expone la metadata completa del modelo (versión, fecha de entrenamiento, hiperparámetros y métricas) sin necesidad de acceder a los archivos internos.

2. **Respuesta enriquecida de /predict**: además de la predicción binaria y la probabilidad, incluye un nivel de riesgo (bajo, moderado, alto, crítico) y una recomendación operativa para el equipo de retención.

## Autor

Carlos [Tu Apellido] — Maestría en Ciencia de Datos e IA, UAGRM
