import pandas as pd
import numpy as np
import random

# Semilla para reproducibilidad
random.seed(42)
np.random.seed(42)

# Definición de estaciones y líneas
estaciones = [
    'Norte', 'Centro', 'Sur', 'Portal Del Quindio', 'El Bosque', 
    'La Clarita', 'Mercado Minorista Quindiano', 'Universidad Del Quindio', 
    'Estadio Centenario', 'Terminal Del Sur'
]

# Líneas con sus estaciones (en orden)
lineas = {
    'A': ['Norte', 'Centro', 'Portal Del Quindio', 'Sur'],
    'B': ['El Bosque', 'Centro', 'Mercado Minorista Quindiano', 'La Clarita'],
    'C': ['Portal Del Quindio', 'Universidad Del Quindio', 'Estadio Centenario', 
          'Mercado Minorista Quindiano', 'Terminal Del Sur']
}

# Tiempo base por estación (minutos)
tiempo_por_estacion = {
    'A': 4,
    'B': 5,
    'C': 3
}

# Función para encontrar la ruta entre dos estaciones
def encontrar_ruta(origen, destino):
    rutas_posibles = []
    
    # Para cada línea, verificar si contiene ambas estaciones
    for linea, estaciones_linea in lineas.items():
        if origen in estaciones_linea and destino in estaciones_linea:
            idx_origen = estaciones_linea.index(origen)
            idx_destino = estaciones_linea.index(destino)
            if idx_origen != idx_destino:
                num_tramos = abs(idx_origen - idx_destino)
                rutas_posibles.append({
                    'linea': linea,
                    'num_tramos': num_tramos,
                    'transbordos': 0,
                    'lineas_usadas': [linea]
                })
    
    # Si no hay ruta directa, buscar con 1 transbordo
    if not rutas_posibles:
        for linea1, estaciones1 in lineas.items():
            if origen in estaciones1:
                for linea2, estaciones2 in lineas.items():
                    if linea1 != linea2 and destino in estaciones2:
                        estaciones_comunes = set(estaciones1) & set(estaciones2)
                        for comun in estaciones_comunes:
                            idx_origen = estaciones1.index(origen)
                            idx_comun1 = estaciones1.index(comun)
                            idx_comun2 = estaciones2.index(comun)
                            idx_destino = estaciones2.index(destino)
                            
                            if idx_origen != idx_comun1 and idx_comun2 != idx_destino:
                                tramos1 = abs(idx_origen - idx_comun1)
                                tramos2 = abs(idx_comun2 - idx_destino)
                                rutas_posibles.append({
                                    'linea': linea1 if tramos1 >= tramos2 else linea2,
                                    'num_tramos': tramos1 + tramos2,
                                    'transbordos': 1,
                                    'lineas_usadas': [linea1, linea2]
                                })
    
    # Seleccionar la ruta con menos transbordos y tramos
    if rutas_posibles:
        mejor_ruta = min(rutas_posibles, key=lambda x: (x['transbordos'], x['num_tramos']))
        return mejor_ruta['linea'], mejor_ruta['num_tramos'], mejor_ruta['transbordos']
    
    return 'A', 2, 1  # Ruta por defecto

# Generar dataset
n_filas = 1000
data = []

for _ in range(n_filas):
    # Seleccionar origen y destino diferentes
    origen, destino = random.sample(estaciones, 2)
    
    # Hora del día (0-23)
    hora = np.random.randint(0, 24)
    
    # Determinar si es hora pico
    es_hora_pico = 1 if (6 <= hora <= 9) or (17 <= hora <= 19) else 0
    
    # Día de semana (0=lunes a 6=domingo)
    dia_semana = np.random.randint(0, 7)
    es_fin_semana = 1 if dia_semana >= 5 else 0
    
    # Calcular ruta óptima
    linea_principal, num_tramos, num_transbordos = encontrar_ruta(origen, destino)
    
    # Calcular tiempo estimado
    tiempo_base = num_tramos * tiempo_por_estacion[linea_principal]
    factor_express = 0.75 if linea_principal == 'C' else 1.0
    tiempo_viaje = tiempo_base * factor_express
    tiempo_espera = 3
    tiempo_transbordo = num_transbordos * 6
    
    if es_hora_pico:
        tiempo_espera *= 1.5
    
    tiempo_estimado = tiempo_viaje + tiempo_espera + tiempo_transbordo
    
    # Tiempo real con variación
    variacion = np.random.uniform(-2, 5)
    tiempo_real = max(1, tiempo_estimado + variacion)
    
    # Ruta óptima
    ruta_optima = 1 if tiempo_real <= tiempo_estimado + 2 else 0
    
    # Pasajeros (mayor en hora pico)
    if es_hora_pico:
        pasajeros = np.random.randint(200, 501)
    else:
        pasajeros = np.random.randint(50, 201)
    
    data.append({
        'estacion_origen': origen,
        'estacion_destino': destino,
        'hora_del_dia': hora,
        'es_hora_pico': es_hora_pico,
        'dia_semana': dia_semana,
        'es_fin_de_semana': es_fin_semana,
        'num_transbordos': num_transbordos,
        'linea_principal': linea_principal,
        'pasajeros_en_sistema': pasajeros,
        'tiempo_estimado_min': round(tiempo_estimado, 2),
        'tiempo_real_min': round(tiempo_real, 2),
        'ruta_optima': ruta_optima
    })

# Crear DataFrame
df = pd.DataFrame(data)

# Guardar a CSV
df.to_csv('dataset_transporte.csv', index=False)

# Mostrar información
print(f"Shape del dataset: {df.shape}")
print("\nPrimeras 5 filas:")
print(df.head())
print("\nEstadísticas básicas:")
print(df.describe())