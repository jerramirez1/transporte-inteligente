"""
inference_engine.py
-------------------
Archivo que contiene el motor de inferencia del sistema experto.

El motor de inferencia es el encargado de revisar las reglas
definidas en la base de conocimiento y aplicar las que
correspondan según la situación actual.

El proceso funciona así:
    1. Recibe los datos del tramo que se está evaluando
    2. Revisa cada regla en orden
    3. Si la condición de la regla se cumple, aplica su efecto
    4. Ese efecto puede modificar el costo o bloquear el tramo
    5. Repite hasta que ninguna regla produzca cambios nuevos

A este proceso se le llama encadenamiento hacia adelante,
porque parte de los hechos conocidos y va construyendo
conclusiones nuevas a medida que las reglas se van disparando.

Por ejemplo, si el contexto indica que es hora pico,
la regla R01 se activa y aumenta el tiempo de espera.
Si además el tramo es de una línea express, también
se activa R02 y reduce el tiempo entre estaciones.
"""

from knowledge_base import RULES, is_segment_closed


def run_inference(context: dict, max_iterations: int = 10) -> dict:
    """
    Ejecuta el motor de inferencia sobre el contexto dado.

    Aplica todas las reglas de RULES de manera iterativa hasta que
    ninguna regla produzca cambios (punto fijo) o se alcance max_iterations.

    Args:
        context       : diccionario con los hechos actuales.
                        Claves comunes:
                          - hora_pico       (bool)
                          - linea_actual    (str)
                          - estacion_actual (str)
                          - cambio_de_linea (bool)
                          - tramo_cerrado   (bool)
                          - num_transbordos (int)
        max_iterations: límite de seguridad para evitar bucles infinitos.

    Retorna:
        dict: el mismo contexto, ahora enriquecido con los efectos de las reglas.
              Incluye la clave "reglas_aplicadas" con la lista de IDs de reglas usadas.
    """
    # Aseguramos que las claves esenciales existan con valores por defecto
    context.setdefault("hora_pico",       False)
    context.setdefault("linea_actual",    "")
    context.setdefault("estacion_actual", "")
    context.setdefault("cambio_de_linea", False)
    context.setdefault("tramo_cerrado",   False)
    context.setdefault("num_transbordos", 0)
    context.setdefault("costo_extra_min", 0)
    context.setdefault("factor_tiempo",   1.0)
    context.setdefault("tramo_invalido",  False)
    context.setdefault("reglas_aplicadas", [])

    reglas_ya_aplicadas: set[str] = set()

    for iteration in range(max_iterations):
        # Tomamos una "foto" del estado antes de aplicar reglas
        estado_anterior = context.copy()

        for rule in RULES:
            # Cada regla se aplica como máximo UNA VEZ por evaluación
            if rule["id"] in reglas_ya_aplicadas:
                continue
            try:
                if rule["condicion"](context):
                    rule["accion"](context)
                    reglas_ya_aplicadas.add(rule["id"])
            except Exception as e:
                # En producción loguearíamos esto con logging.warning(...)
                print(f"[Motor] Advertencia al evaluar regla {rule['id']}: {e}")

        # Si el contexto no cambió, llegamos al punto fijo → detener
        if context == estado_anterior:
            break

    return context


def evaluate_segment(
    origin: str,
    destination: str,
    line: str,
    nominal_time: int,
    hora_pico: bool = False,
    linea_anterior: str = "",
) -> dict:
    """
    Evalúa un tramo individual entre dos estaciones y retorna su costo real.

    Esta función es llamada por el planificador (route_planner.py)
    para cada arista del grafo durante la búsqueda.

    Args:
        origin        : nombre de la estación de origen del tramo
        destination   : nombre de la estación de destino del tramo
        line          : nombre de la línea del tramo
        nominal_time  : tiempo base en minutos (del grafo)
        hora_pico     : True si es hora pico
        linea_anterior: línea desde la que venía el pasajero (para detectar transbordo)

    Retorna:
        dict con:
          - costo_real_min  : tiempo real ajustado por reglas (float)
          - invalido        : True si el tramo no puede usarse
          - reglas_aplicadas: lista de IDs de reglas que se dispararon
          - detalle         : descripción textual del costo
    """
    cambio_de_linea = bool(linea_anterior and linea_anterior != line)

    context = {
        "hora_pico":       hora_pico,
        "linea_actual":    line,
        "estacion_actual": origin,
        "cambio_de_linea": cambio_de_linea,
        "tramo_cerrado":   is_segment_closed(origin, destination),
        "num_transbordos": 1 if cambio_de_linea else 0,
        "wait_time_min":   3,     # valor por defecto (Regla R01 puede ajustarlo)
    }

    # Ejecutar el motor de inferencia
    context = run_inference(context)

    # Si el tramo es inválido (cerrado, etc.), retornar inmediatamente
    if context.get("tramo_invalido", False):
        return {
            "costo_real_min":   float("inf"),
            "invalido":         True,
            "reglas_aplicadas": context["reglas_aplicadas"],
            "detalle":          f"Tramo {origin}→{destination} no disponible",
        }

    # Cálculo del costo real:
    #   tiempo_base × factor_tiempo (express) + costo_extra (transbordo) + espera
    costo = (
        nominal_time * context.get("factor_tiempo", 1.0)
        + context.get("costo_extra_min", 0)
        + context.get("wait_time_min", 3)
    )
    costo = max(costo, 0)  # nunca negativo

    return {
        "costo_real_min":   round(costo, 2),
        "invalido":         False,
        "reglas_aplicadas": context["reglas_aplicadas"],
        "detalle": (
            f"{nominal_time} min base × {context.get('factor_tiempo',1.0)} "
            f"+ {context.get('costo_extra_min',0)} transbordo "
            f"+ {context.get('wait_time_min',3)} espera"
        ),
    }


# ---------------------------------------------------------------------------
# Bloque de prueba rápida
# Ejecuta: python inference_engine.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Prueba del motor de inferencia ===\n")

    # Caso 1: Tramo normal, sin hora pico
    resultado = evaluate_segment(
        origin="Norte", destination="Centro",
        line="A", nominal_time=4,
        hora_pico=False, linea_anterior=""
    )
    print(f"Caso 1 — Normal:")
    print(f"  Costo real : {resultado['costo_real_min']} min")
    print(f"  Reglas     : {resultado['reglas_aplicadas']}")
    print(f"  Detalle    : {resultado['detalle']}\n")

    # Caso 2: Tramo con hora pico + transbordo
    resultado = evaluate_segment(
        origin="Centro", destination="Mercado Minorista Quindiano",
        line="B", nominal_time=5,
        hora_pico=True, linea_anterior="A"
    )
    print(f"Caso 2 — Hora pico + transbordo (A→B en Centro):")
    print(f"  Costo real : {resultado['costo_real_min']} min")
    print(f"  Reglas     : {resultado['reglas_aplicadas']}")
    print(f"  Detalle    : {resultado['detalle']}\n")

    # Caso 3: Línea express
    resultado = evaluate_segment(
        origin="Portal del Quindio", destination="Universidad Del Quindio",
        line="C", nominal_time=3,
        hora_pico=False, linea_anterior="A"
    )
    print(f"Caso 3 — Línea express + transbordo (A→C en Portal del Quindio):")
    print(f"  Costo real : {resultado['costo_real_min']} min")
    print(f"  Reglas     : {resultado['reglas_aplicadas']}")
    print(f"  Detalle    : {resultado['detalle']}\n")
