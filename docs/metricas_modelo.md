# Reporte de métricas — modelo_churn_v1

**Fecha de entrenamiento:** 2026-06-17T20:24:29.511797+00:00
**Algoritmo:** RandomForestClassifier (n_estimators=100, max_depth=5)

## Métricas de evaluación (conjunto test: 200 registros)

| Métrica   | Valor  |
|-----------|--------|
| Accuracy  | 0.8550 |
| Precision | 0.8571 |
| Recall    | 0.1765 |
| F1-Score  | 0.2927 |

## Matriz de confusión

|                | Predicho: No churn | Predicho: Churn |
|----------------|-------------------:|----------------:|
| Real: No churn | 165         | 1      |
| Real: Churn    | 28         | 6      |

## Variables de entrada

- edad
- antiguedad_meses
- saldo_promedio
- reclamos
- usa_app
