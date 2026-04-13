"""
dataset_generator_unsupervised.py
---------------------------------
Genera dataset para aprendizaje NO SUPERVISADO del sistema de transporte masivo.
Este dataset NO incluye etiquetas (no tiene tiempo_real_min ni ruta_optima).
"""

import pandas as pd
import numpy as np
import random

# Configuración
random.seed(42)
np.random.seed(42)

# Estaciones del sistema
estaciones = [
    'Norte', 'Centro', 'Sur', 'Portal Del Quindio', 'El Bosque',
    'La Clarita', 'Mercado Minorista Quindiano', 'Universidad Del Quindio',
    'Estadio Centenario', 'Terminal Del Sur'
]

def generar_dataset(n_viajes: int = 1000, archivo_salida: str = "dataset_transporte_unsupervised.csv"):
    """
    Genera dataset sin etiquetas para aprendizaje no supervisado.
    
    Args:
        n_viajes: Número de viajes a generar
        archivo_salida: Nombre del archivo CSV de salida
    """
    data = []
    
    for _ in range(n_viajes):
        # Seleccionar origen y destino diferentes
        origen = random.choice(estaciones)
        destino = random.choice([e for e in estaciones if e != origen])
        
        # Hora del día (0-23)
        hora = np.random.randint(0, 24)
        
        # Día de semana (0=lunes a 6=domingo)
        dia_semana = np.random.randint(0, 7)
        
        # Número de transbordos (0, 1 o 2)
        if random.random() < 0.6:
            transbordos = 0
        elif random.random() < 0.85:
            transbordos = 1
        else:
            transbordos = 2
        
        # Simular pasajeros (mayor en hora pico)
        es_pico = 1 if (6 <= hora <= 9) or (17 <= hora <= 19) else 0
        if es_pico:
            pasajeros = np.random.randint(200, 500)
        else:
            pasajeros = np.random.randint(50, 200)
        
        # Tiempo estimado (solo para análisis, no es etiqueta objetivo)
        tiempo_estimado = 10 + transbordos * 6 + np.random.randint(0, 15)
        
        data.append({
            'estacion_origen': origen,
            'estacion_destino': destino,
            'hora_del_dia': hora,
            'dia_semana': dia_semana,
            'num_transbordos': transbordos,
            'pasajeros_en_sistema': pasajeros,
            'tiempo_estimado_min': tiempo_estimado
        })
    
    df = pd.DataFrame(data)
    df.to_csv(archivo_salida, index=False)
    
    print(f"✅ Dataset generado: {archivo_salida}")
    print(f"   - Filas: {len(df)}")
    print(f"   - Columnas: {list(df.columns)}")
    print(f"\nPrimeras 5 filas:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    generar_dataset()