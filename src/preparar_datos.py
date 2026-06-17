"""
preparar_datos.py
Genera un dataset sintético de churn y lo divide en train/test.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
N_MUESTRAS = 1000
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def generar_dataset(n: int = N_MUESTRAS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    edad = rng.randint(18, 70, size=n)
    antiguedad_meses = rng.randint(1, 72, size=n)
    saldo_promedio = rng.uniform(100, 5000, size=n).round(2)
    reclamos = rng.poisson(lam=1.5, size=n)
    usa_app = rng.binomial(1, 0.6, size=n)

    score = (
        -0.02 * antiguedad_meses
        + 0.3 * reclamos
        - 0.4 * usa_app
        + 0.01 * (70 - edad)
        - 0.0002 * saldo_promedio
        + rng.normal(0, 0.5, size=n)
    )
    probabilidad = 1 / (1 + np.exp(-score))
    churn = (probabilidad >= 0.5).astype(int)

    return pd.DataFrame({
        "edad": edad,
        "antiguedad_meses": antiguedad_meses,
        "saldo_promedio": saldo_promedio,
        "reclamos": reclamos,
        "usa_app": usa_app,
        "churn": churn,
    })


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Generando dataset de churn...")
    df = generar_dataset()

    ruta = os.path.join(DATA_DIR, "churn_clientes.csv")
    df.to_csv(ruta, index=False)
    print(f"  Dataset completo: {ruta}  ({len(df)} registros)")

    train, test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["churn"]
    )
    train.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    print(f"  Train: {len(train)} registros  |  Test: {len(test)} registros")
    print("Preparacion de datos completada.")


if __name__ == "__main__":
    main()
