"""
ml_model.py
-----------
Modelos de aprendizaje automático para el sistema de transporte masivo.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def cargar_datos(ruta_csv: str = "dataset_transporte.csv") -> pd.DataFrame:
    """Carga el dataset y prepara las columnas categóricas."""
    df = pd.read_csv(ruta_csv)

    le_estacion = LabelEncoder()
    le_linea = LabelEncoder()

    todas_estaciones = pd.concat([df["estacion_origen"], df["estacion_destino"]])
    le_estacion.fit(todas_estaciones)

    df["estacion_origen_enc"] = le_estacion.transform(df["estacion_origen"])
    df["estacion_destino_enc"] = le_estacion.transform(df["estacion_destino"])
    df["linea_principal_enc"] = le_linea.fit_transform(df["linea_principal"])

    return df, le_estacion, le_linea


def preparar_features(df: pd.DataFrame):
    """Prepara variables de entrada (X) y salida (y)."""
    features = [
        "hora_del_dia",
        "es_hora_pico",
        "dia_semana",
        "es_fin_de_semana",
        "num_transbordos",
        "pasajeros_en_sistema",
        "estacion_origen_enc",
        "estacion_destino_enc",
        "linea_principal_enc",
    ]

    X = df[features]
    y1 = df["tiempo_real_min"]
    y2 = df["ruta_optima"]

    return X, y1, y2


def entrenar_modelo_regresion(X_train, X_test, y_train, y_test):
    """Entrena Random Forest Regressor para predecir tiempo real."""
    print("\n" + "=" * 55)
    print("  MODELO 1 — REGRESIÓN: Predicción de tiempo real")
    print("=" * 55)

    modelo = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10,
    )

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n  Resultados en datos de prueba:")
    print(f"  MAE  : {mae:.2f} minutos")
    print(f"  RMSE : {rmse:.2f} minutos")
    print(f"  R²   : {r2:.4f}")

    print(f"\n  Variables más importantes:")
    importancias = pd.Series(
        modelo.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    for var, imp in importancias.items():
        barra = "█" * int(imp * 50)
        print(f"  {var:25s} {barra} {imp:.4f}")

    return modelo


def entrenar_modelo_clasificacion(X_train, X_test, y_train, y_test):
    """Entrena Random Forest Classifier para predecir ruta óptima."""
    print("\n" + "=" * 55)
    print("  MODELO 2 — CLASIFICACIÓN: Predicción de ruta óptima")
    print("=" * 55)

    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
    )

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  Resultados en datos de prueba:")
    print(f"  Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")

    print(f"\n  Reporte detallado:")
    print(
        classification_report(
            y_test, y_pred, target_names=["No óptima (0)", "Óptima (1)"]
        )
    )

    print(f"  Matriz de confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {'':20s} Predicho: No óptima  Predicho: Óptima")
    print(f"  Real: No óptima  {cm[0][0]:>18}  {cm[0][1]:>15}")
    print(f"  Real: Óptima     {cm[1][0]:>18}  {cm[1][1]:>15}")

    print(f"\n  Variables más importantes:")
    importancias = pd.Series(
        modelo.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    for var, imp in importancias.items():
        barra = "█" * int(imp * 50)
        print(f"  {var:25s} {barra} {imp:.4f}")

    return modelo


def predecir_viaje(
    modelo_regresion,
    modelo_clasificacion,
    le_estacion,
    le_linea,
    origen: str,
    destino: str,
    hora: int,
    dia_semana: int,
    num_transbordos: int,
    pasajeros: int,
    linea: str,
):
    """Predice tiempo real y optimalidad para un viaje nuevo."""
    print("\n" + "=" * 55)
    print("  PREDICCIÓN DE VIAJE NUEVO")
    print("=" * 55)
    print(f"  Origen      : {origen}")
    print(f"  Destino     : {destino}")
    print(f"  Hora        : {hora}:00")
    print(f"  Día semana  : {dia_semana} (0=Lunes)")
    print(f"  Transbordos : {num_transbordos}")
    print(f"  Pasajeros   : {pasajeros}")

    es_hora_pico = 1 if (6 <= hora <= 9) or (17 <= hora <= 19) else 0
    es_fin_semana = 1 if dia_semana >= 5 else 0

    origen_enc = le_estacion.transform([origen])[0]
    destino_enc = le_estacion.transform([destino])[0]
    linea_enc = le_linea.transform([linea])[0]

    X_nuevo = pd.DataFrame(
        [
            {
                "hora_del_dia": hora,
                "es_hora_pico": es_hora_pico,
                "dia_semana": dia_semana,
                "es_fin_de_semana": es_fin_semana,
                "num_transbordos": num_transbordos,
                "pasajeros_en_sistema": pasajeros,
                "estacion_origen_enc": origen_enc,
                "estacion_destino_enc": destino_enc,
                "linea_principal_enc": linea_enc,
            }
        ]
    )

    tiempo_predicho = modelo_regresion.predict(X_nuevo)[0]
    optima_predicha = modelo_clasificacion.predict(X_nuevo)[0]

    print(f"\n  Tiempo real predicho : {tiempo_predicho:.2f} minutos")
    print(f"  ¿Ruta óptima?        : {'Sí ✓' if optima_predicha == 1 else 'No ✗'}")
    print("=" * 55)


if __name__ == "__main__":
    print("\n  Cargando dataset...")
    df, le_estacion, le_linea = cargar_datos("dataset_transporte.csv")
    print(f"  Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")

    X, y1, y2 = preparar_features(df)

    X_train, X_test, y1_train, y1_test = train_test_split(
        X, y1, test_size=0.2, random_state=42
    )
    _, _, y2_train, y2_test = train_test_split(
        X, y2, test_size=0.2, random_state=42
    )

    print(f"  Entrenamiento: {X_train.shape[0]} filas")
    print(f"  Prueba       : {X_test.shape[0]} filas")

    modelo_reg = entrenar_modelo_regresion(X_train, X_test, y1_train, y1_test)
    modelo_clas = entrenar_modelo_clasificacion(X_train, X_test, y2_train, y2_test)

    # Ejemplos de predicción
    print("\n" + " EJEMPLOS DE PREDICCIÓN ".center(55, "="))
    
    # Ejemplo 1: Viaje en hora pico
    predecir_viaje(
        modelo_reg,
        modelo_clas,
        le_estacion,
        le_linea,
        origen="Norte",
        destino="Terminal Del Sur",
        hora=8,
        dia_semana=0,
        num_transbordos=2,
        pasajeros=350,
        linea="C",
    )
    
    # Ejemplo 2: Viaje en hora valle
    predecir_viaje(
        modelo_reg,
        modelo_clas,
        le_estacion,
        le_linea,
        origen="Centro",
        destino="El Bosque",
        hora=14,
        dia_semana=2,
        num_transbordos=0,
        pasajeros=80,
        linea="B",
    )
    
    # Ejemplo 3: Viaje fin de semana
    predecir_viaje(
        modelo_reg,
        modelo_clas,
        le_estacion,
        le_linea,
        origen="Universidad Del Quindio",
        destino="Mercado Minorista Quindiano",
        hora=11,
        dia_semana=6,
        num_transbordos=1,
        pasajeros=120,
        linea="C",
    )