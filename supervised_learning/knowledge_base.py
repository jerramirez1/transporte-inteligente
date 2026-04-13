"""
knowledge_base.py
-----------------
Base de conocimiento del sistema experto de transporte masivo.

CONCEPTO CLAVE para el junior:
  Una base de conocimiento tiene dos partes:
  1. HECHOS     → cosas que sabemos que son verdad (datos del grafo)
  2. REGLAS     → relaciones del tipo "SI ... ENTONCES ..."

  Ejemplo de regla humana:
    SI la línea está en hora pico Y el tramo es concurrido
    ENTONCES el tiempo real es mayor al tiempo nominal

  Aquí lo representamos en Python con diccionarios y listas.
"""

# ---------------------------------------------------------------------------
# SECCIÓN 1: Hechos base — atributos de líneas y estaciones
# ---------------------------------------------------------------------------

# Cada estación tiene: nombre, línea(s) a la que pertenece, y si es terminal
STATIONS: dict[str, dict] = {
    "Centro":        {"lineas": ["A", "B"], "es_intercambio": True,  "es_terminal": False},
    "Norte":          {"lineas": ["A"],       "es_intercambio": False, "es_terminal": True},
    "Sur":            {"lineas": ["A"],       "es_intercambio": False, "es_terminal": True},
    "La Clarita":        {"lineas": ["B"],       "es_intercambio": False, "es_terminal": True},
    "El Bosque":      {"lineas": ["B"],       "es_intercambio": False, "es_terminal": True},
    "Portal Del Quindio":    {"lineas": ["A", "C"],  "es_intercambio": True,  "es_terminal": False},
    "Universidad Del Quindio":    {"lineas": ["C"],       "es_intercambio": False, "es_terminal": False},
    "Estadio Centenario":        {"lineas": ["C"],       "es_intercambio": False, "es_terminal": False},
    "Terminal Del Sur":   {"lineas": ["C"],       "es_intercambio": False, "es_terminal": True},
    "Mercado Minorista Quindiano":        {"lineas": ["B", "C"],  "es_intercambio": True,  "es_terminal": False},
}

# Cada línea tiene: tiempo entre estaciones (minutos), y si es rápida (express)
LINES: dict[str, dict] = {
    "A": {"tiempo_base_min": 4, "es_express": False, "capacidad": "alta"},
    "B": {"tiempo_base_min": 5, "es_express": False, "capacidad": "media"},
    "C": {"tiempo_base_min": 3, "es_express": True,  "capacidad": "baja"},
}

# Tiempo en minutos que toma hacer un transbordo en una estación de intercambio
TRANSFER_TIME_MIN: int = 6

# Penalización por subir al primer tren (espera promedio)
WAIT_TIME_MIN: int = 3


# ---------------------------------------------------------------------------
# SECCIÓN 2: Reglas lógicas
# ---------------------------------------------------------------------------
# Representamos cada regla como un diccionario con:
#   - "descripcion" : texto legible para humanos
#   - "condicion"   : función lambda que recibe contexto y retorna True/False
#   - "accion"      : función lambda que transforma/anota el contexto
#
# El motor de inferencia (inference_engine.py) leerá esta lista
# y aplicará las reglas que apliquen al contexto actual.

RULES: list[dict] = [

    # -----------------------------------------------------------------------
    # REGLA 1: Hora pico — aumenta el tiempo de espera
    # SI es hora pico ENTONCES penalizar tiempo de espera
    # -----------------------------------------------------------------------
    {
        "id": "R01",
        "descripcion": "En hora pico el tiempo de espera aumenta un 50%",
        "condicion": lambda ctx: ctx.get("hora_pico", False),
        "accion": lambda ctx: ctx.update({
            "wait_time_min": int(WAIT_TIME_MIN * 1.5),
            "reglas_aplicadas": ctx.get("reglas_aplicadas", []) + ["R01"]
        }),
    },

    # -----------------------------------------------------------------------
    # REGLA 2: Línea express — reduce tiempo entre estaciones
    # SI la línea del tramo es express ENTONCES usar tiempo reducido
    # -----------------------------------------------------------------------
    {
        "id": "R02",
        "descripcion": "Las líneas express reducen el tiempo entre estaciones un 25%",
        "condicion": lambda ctx: LINES.get(ctx.get("linea_actual", ""), {}).get("es_express", False),
        "accion": lambda ctx: ctx.update({
            "factor_tiempo": 0.75,
            "reglas_aplicadas": ctx.get("reglas_aplicadas", []) + ["R02"]
        }),
    },

    # -----------------------------------------------------------------------
    # REGLA 3: Estación de intercambio — agrega tiempo de transbordo
    # SI la estación es de intercambio Y se cambia de línea
    # ENTONCES agregar TRANSFER_TIME_MIN al costo
    # -----------------------------------------------------------------------
    {
        "id": "R03",
        "descripcion": "El transbordo en estación de intercambio agrega tiempo",
        "condicion": lambda ctx: (
            STATIONS.get(ctx.get("estacion_actual", ""), {}).get("es_intercambio", False)
            and ctx.get("cambio_de_linea", False)
        ),
        "accion": lambda ctx: ctx.update({
            "costo_extra_min": ctx.get("costo_extra_min", 0) + TRANSFER_TIME_MIN,
            "reglas_aplicadas": ctx.get("reglas_aplicadas", []) + ["R03"]
        }),
    },

    # -----------------------------------------------------------------------
    # REGLA 4: Tramo cerrado — bloquear ese tramo
    # SI el tramo está reportado como cerrado ENTONCES no se puede usar
    # -----------------------------------------------------------------------
    {
        "id": "R04",
        "descripcion": "Un tramo cerrado no puede ser usado en la ruta",
        "condicion": lambda ctx: ctx.get("tramo_cerrado", False),
        "accion": lambda ctx: ctx.update({
            "tramo_invalido": True,
            "reglas_aplicadas": ctx.get("reglas_aplicadas", []) + ["R04"]
        }),
    },

    # -----------------------------------------------------------------------
    # REGLA 5: Preferencia de ruta directa
    # SI la ruta no tiene transbordos ENTONCES reducir costo total (bonificación)
    # -----------------------------------------------------------------------
    {
        "id": "R05",
        "descripcion": "Rutas sin transbordo tienen prioridad (descuento de costo)",
        "condicion": lambda ctx: ctx.get("num_transbordos", 1) == 0,
        "accion": lambda ctx: ctx.update({
            "costo_extra_min": ctx.get("costo_extra_min", 0) - 2,
            "reglas_aplicadas": ctx.get("reglas_aplicadas", []) + ["R05"]
        }),
    },
]


# ---------------------------------------------------------------------------
# SECCIÓN 3: Tramos cerrados (se podrían cargar desde una API en producción)
# ---------------------------------------------------------------------------
# Lista de tuplas (estacion_origen, estacion_destino) que están fuera de servicio
CLOSED_SEGMENTS: list[tuple[str, str]] = [
    # Ejemplo: ("Norte", "Central"),  # Descomentar para simular cierre
]


# ---------------------------------------------------------------------------
# Función utilitaria — consultada por otros módulos
# ---------------------------------------------------------------------------

def is_segment_closed(origin: str, destination: str) -> bool:
    """Retorna True si el tramo entre origin y destination está cerrado."""
    return (
        (origin, destination) in CLOSED_SEGMENTS
        or (destination, origin) in CLOSED_SEGMENTS
    )


def get_station_info(station_name: str) -> dict:
    """Retorna el diccionario de atributos de una estación, o {} si no existe."""
    return STATIONS.get(station_name, {})


def get_line_info(line_name: str) -> dict:
    """Retorna el diccionario de atributos de una línea, o {} si no existe."""
    return LINES.get(line_name, {})
