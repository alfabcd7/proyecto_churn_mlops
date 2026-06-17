"""
entrenar_modelo.py
Entrena un RandomForest, lo serializa como .joblib,
genera metadata JSON y reporte de métricas.
"""

import os
import json
from datetime import datetime, timezone

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")

FEATURES = ["edad", "antiguedad_meses", "saldo_promedio", "reclamos", "usa_app"]
TARGET = "churn"
MODEL_VERSION = "v1"


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    # ── Cargar datos ──
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    # ── Entrenar modelo ──
    print("Entrenando modelo RandomForest...")
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=SEED,
    )
    modelo.fit(X_train, y_train)

    # ── Evaluar ──
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # ── 1. Serializar modelo (.joblib) ──
    ruta_modelo = os.path.join(MODELS_DIR, f"modelo_churn_{MODEL_VERSION}.joblib")
    joblib.dump(modelo, ruta_modelo)
    print(f"  Modelo guardado: {ruta_modelo}")

    # ── 2. Metadata JSON ──
    metadata = {
        "modelo": "RandomForestClassifier",
        "version": MODEL_VERSION,
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(),
        "hiperparametros": {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": SEED,
        },
        "features": FEATURES,
        "target": TARGET,
        "metricas": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        },
        "datos": {
            "total_registros": len(train) + len(test),
            "registros_train": len(train),
            "registros_test": len(test),
        },
        "matriz_confusion": {
            "TN": cm[0][0], "FP": cm[0][1],
            "FN": cm[1][0], "TP": cm[1][1],
        },
    }

    ruta_metadata = os.path.join(MODELS_DIR, f"modelo_churn_{MODEL_VERSION}_metadata.json")
    with open(ruta_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Metadata guardada: {ruta_metadata}")

    # ── 3. Reporte de métricas (Markdown) ──
    reporte = f"""# Reporte de métricas — modelo_churn_{MODEL_VERSION}

**Fecha de entrenamiento:** {metadata["fecha_entrenamiento"]}
**Algoritmo:** RandomForestClassifier (n_estimators=100, max_depth=5)

## Métricas de evaluación (conjunto test: {len(test)} registros)

| Métrica   | Valor  |
|-----------|--------|
| Accuracy  | {accuracy:.4f} |
| Precision | {precision:.4f} |
| Recall    | {recall:.4f} |
| F1-Score  | {f1:.4f} |

## Matriz de confusión

|                | Predicho: No churn | Predicho: Churn |
|----------------|-------------------:|----------------:|
| Real: No churn | {cm[0][0]}         | {cm[0][1]}      |
| Real: Churn    | {cm[1][0]}         | {cm[1][1]}      |

## Variables de entrada

{chr(10).join(f"- {feat}" for feat in FEATURES)}
"""

    ruta_metricas = os.path.join(DOCS_DIR, "metricas_modelo.md")
    with open(ruta_metricas, "w", encoding="utf-8") as f:
        f.write(reporte)
    print(f"  Metricas guardadas: {ruta_metricas}")
    print("Entrenamiento y evaluacion completados.")


if __name__ == "__main__":
    main()
