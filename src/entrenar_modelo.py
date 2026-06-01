from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TRAIN_DATA = DATA_DIR / "train.csv"
TEST_DATA = DATA_DIR / "test.csv"
MODEL_FILE = MODELS_DIR / "modelo_churn.pkl"


def entrenar_modelo():
    """
    Entrena y compara dos modelos de clasificación para predecir churn:
      - Logistic Regression (modelo original)
      - Random Forest (segundo algoritmo — experimento)

    Cambios respecto a la versión original:
      1. Se agrega RandomForestClassifier como segundo algoritmo.
      2. Se modifican hiperparámetros: n_estimators=50, max_depth=3.
      3. Se agrega ROC-AUC como métrica adicional de comparacion.
    """
    if not TRAIN_DATA.exists():
        raise FileNotFoundError(
            "No se encontró data/train.csv. Primero ejecuta src/preparar_datos.py"
        )

    MODELS_DIR.mkdir(exist_ok=True)

    train_df = pd.read_csv(TRAIN_DATA)
    test_df  = pd.read_csv(TEST_DATA)

    X_train = train_df.drop(columns=["churn"])
    y_train = train_df["churn"]
    X_test  = test_df.drop(columns=["churn"])
    y_test  = test_df["churn"]

    # Modelo 1: Logistic Regression (baseline original)
    modelo_lr = Pipeline(
        steps=[
            ("escalado", StandardScaler()),
            ("clasificador", LogisticRegression()),
        ]
    )
    modelo_lr.fit(X_train, y_train)
    pred_lr  = modelo_lr.predict(X_test)
    proba_lr = modelo_lr.predict_proba(X_test)[:, 1]

    # Modelo 2: Random Forest (segundo algoritmo)
    modelo_rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=42,
    )
    modelo_rf.fit(X_train, y_train)
    pred_rf  = modelo_rf.predict(X_test)
    proba_rf = modelo_rf.predict_proba(X_test)[:, 1]

    # Comparativa de metricas
    print("=" * 55)
    print("  COMPARATIVA DE MODELOS — experimento-randomforest")
    print("=" * 55)
    print(f"  {'Metrica':<14} {'Log. Regression':>16} {'Random Forest':>14}")
    print("  " + "-" * 47)

    metricas = {
        "Accuracy":  (accuracy_score(y_test, pred_lr),
                      accuracy_score(y_test, pred_rf)),
        "Precision": (precision_score(y_test, pred_lr, zero_division=0),
                      precision_score(y_test, pred_rf, zero_division=0)),
        "Recall":    (recall_score(y_test, pred_lr, zero_division=0),
                      recall_score(y_test, pred_rf, zero_division=0)),
        "F1-score":  (f1_score(y_test, pred_lr, zero_division=0),
                      f1_score(y_test, pred_rf, zero_division=0)),
        "ROC-AUC":   (roc_auc_score(y_test, proba_lr),
                      roc_auc_score(y_test, proba_rf)),
    }

    for nombre, (v_lr, v_rf) in metricas.items():
        mejor = "<-- MEJOR" if v_rf >= v_lr else ""
        print(f"  {nombre:<14} {v_lr:>16.4f} {v_rf:>14.4f}  {mejor}")

    print("  " + "-" * 47)

    # Seleccion del mejor modelo por F1
    f1_lr = metricas["F1-score"][0]
    f1_rf = metricas["F1-score"][1]

    if f1_rf >= f1_lr:
        mejor_modelo  = modelo_rf
        nombre_ganador = "RandomForest"
    else:
        mejor_modelo  = modelo_lr
        nombre_ganador = "LogisticRegression"

    joblib.dump(mejor_modelo, MODEL_FILE)

    print(f"\n  Mejor modelo por F1-score: {nombre_ganador}")
    print(f"  Modelo guardado en: {MODEL_FILE}")


if __name__ == "__main__":
    entrenar_modelo()
