"""
route_planner.py
----------------
Archivo encargado de encontrar la mejor ruta entre dos estaciones.

Para buscar la ruta uso el algoritmo A* (A-estrella), que es más
eficiente que otros algoritmos como Dijkstra porque no explora
todas las rutas posibles, sino que usa una estimación (heurística)
para ir directo hacia el destino.

La heurística que usé es:
    h(n) = estaciones restantes × tiempo mínimo por estación

Esta estimación nunca exagera el costo real, lo que garantiza
que la ruta encontrada siempre sea la óptima.

El costo de cada tramo se calcula así:
    f(n) = g(n) + h(n)

    g(n) = tiempo real acumulado hasta la estación actual
    h(n) = estimación del tiempo que falta hasta el destino

Cada vez que el algoritmo evalúa un tramo entre dos estaciones,
consulta al motor de inferencia para obtener el costo real,
teniendo en cuenta factores como hora pico, líneas express
y tiempos de transbordo definidos en las reglas.
"""

import heapq
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from transport_graph import build_graph, get_neighbors
from inference_engine import evaluate_segment
from knowledge_base import LINES


# Estructura de datos para un nodo en la cola de prioridad de A*

@dataclass(order=True)
class SearchNode:
    """
    Representa un estado en la búsqueda A*.
    
    `order=True` permite que Python compare dos SearchNode por f_score,
    necesario para la cola de prioridad (heapq).
    """
    f_score:       float               # f(n) = g + h — usado para ordenar
    g_score:       float = field(compare=False)  # costo real acumulado
    station:       str   = field(compare=False)  # estación actual
    linea_actual:  str   = field(compare=False)  # línea desde la que llegamos
    path:          list  = field(compare=False, default_factory=list)  # ruta hasta aquí
    path_detail:   list  = field(compare=False, default_factory=list)  # detalles de cada paso


# Heurística admisible para A*


def heuristic(G: nx.DiGraph, current: str, goal: str) -> float:
    """
    Estima el costo mínimo desde `current` hasta `goal`.
    
    Estrategia: contamos cuántas estaciones hay en el camino más corto
    (en términos de número de saltos, sin pesos) y multiplicamos por
    el tiempo mínimo posible entre estaciones en todo el sistema.
    
    Esto NUNCA sobreestima el costo real → heurística admisible → A* óptimo.

    Args:
        G       : el grafo de transporte
        current : nombre de la estación actual
        goal    : nombre de la estación destino

    Retorna:
        float: estimación del costo restante en minutos
    """
    try:
        # Número de saltos en el camino más corto (sin pesos)
        saltos = nx.shortest_path_length(G, current, goal)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")

    # Tiempo mínimo real de cualquier línea en el sistema
    tiempo_minimo = min(
        info["tiempo_base_min"] * (0.75 if info.get("es_express") else 1.0)
        for info in LINES.values()
    )

    return saltos * tiempo_minimo


# ---------------------------------------------------------------------------
# Algoritmo A* principal
# ---------------------------------------------------------------------------

def find_best_route(
    origin: str,
    destination: str,
    hora_pico: bool = False,
    G: Optional[nx.DiGraph] = None,
) -> dict:
    """
    Encuentra la ruta óptima desde `origin` hasta `destination`.

    Combina A* con el motor de inferencia para calcular costos reales.

    Args:
        origin      : nombre de la estación de partida
        destination : nombre de la estación de llegada
        hora_pico   : True si la búsqueda ocurre en hora pico
        G           : grafo (se construye automáticamente si no se pasa)

    Retorna:
        dict con:
          - encontrada      (bool)   : True si se halló una ruta
          - ruta            (list)   : lista de nombres de estaciones
          - tiempo_total    (float)  : minutos totales estimados
          - num_transbordos (int)    : cantidad de cambios de línea
          - pasos           (list)   : instrucciones detalladas paso a paso
          - reglas_usadas   (set)    : IDs de todas las reglas que se dispararon
          - mensaje         (str)    : descripción legible de la ruta
    """
    if G is None:
        G = build_graph()

    # Validar que las estaciones existan en el grafo
    if origin not in G.nodes:
        return {"encontrada": False, "mensaje": f"Estación '{origin}' no existe en el sistema."}
    if destination not in G.nodes:
        return {"encontrada": False, "mensaje": f"Estación '{destination}' no existe en el sistema."}
    if origin == destination:
        return {
            "encontrada": True, "ruta": [origin], "tiempo_total": 0,
            "num_transbordos": 0, "pasos": [], "reglas_usadas": set(),
            "mensaje": "Ya estás en tu destino."
        }

    # Cola de prioridad (min-heap): menor f_score primero
    # Empezamos en el origen, sin línea previa, costo 0
    open_list: list[SearchNode] = []
    inicio = SearchNode(
        f_score=heuristic(G, origin, destination),
        g_score=0.0,
        station=origin,
        linea_actual="",
        path=[origin],
        path_detail=[],
    )
    heapq.heappush(open_list, inicio)

    # Registro de la mejor g_score que hemos encontrado para cada estación
    # Clave: (estacion, linea_actual) para distinguir el costo según desde
    # qué línea llegamos (afecta transbordos)
    best_g: dict[tuple, float] = {(origin, ""): 0.0}

    reglas_usadas_global: set[str] = set()

    while open_list:
        current_node = heapq.heappop(open_list)

        # ¿Llegamos al destino?
        if current_node.station == destination:
            # Calcular número de transbordos: cuántas veces cambia la línea en la ruta
            transbordos = sum(
                1 for paso in current_node.path_detail
                if paso.get("es_transbordo", False)
            )
            return {
                "encontrada":      True,
                "ruta":            current_node.path,
                "tiempo_total":    round(current_node.g_score, 2),
                "num_transbordos": transbordos,
                "pasos":           current_node.path_detail,
                "reglas_usadas":   reglas_usadas_global,
                "mensaje":         _format_route_message(current_node.path_detail, current_node.g_score),
            }

        # Expandir vecinos
        for neighbor_info in get_neighbors(G, current_node.station):
            vecino       = neighbor_info["estacion"]
            linea        = neighbor_info["linea"]
            tiempo_base  = neighbor_info["tiempo"]

            # Evaluar el tramo con el motor de inferencia
            eval_result = evaluate_segment(
                origin         = current_node.station,
                destination    = vecino,
                line           = linea,
                nominal_time   = tiempo_base,
                hora_pico      = hora_pico,
                linea_anterior = current_node.linea_actual,
            )

            # Recopilar qué reglas se usaron en esta evaluación
            reglas_usadas_global.update(eval_result.get("reglas_aplicadas", []))

            # Si el tramo está bloqueado por alguna regla, saltar
            if eval_result.get("invalido", False):
                continue

            nuevo_g = current_node.g_score + eval_result["costo_real_min"]

            # Si ya encontramos una forma más barata de llegar aquí, ignorar
            state_key = (vecino, linea)
            if nuevo_g >= best_g.get(state_key, float("inf")):
                continue

            best_g[state_key] = nuevo_g

            es_transbordo = bool(current_node.linea_actual and current_node.linea_actual != linea)

            # Construir el registro de este paso
            paso = {
                "desde":         current_node.station,
                "hasta":         vecino,
                "linea":         linea,
                "tiempo":        eval_result["costo_real_min"],
                "es_transbordo": es_transbordo,
                "detalle":       eval_result["detalle"],
            }

            nuevo_nodo = SearchNode(
                f_score      = nuevo_g + heuristic(G, vecino, destination),
                g_score      = nuevo_g,
                station      = vecino,
                linea_actual = linea,
                path         = current_node.path + [vecino],
                path_detail  = current_node.path_detail + [paso],
            )
            heapq.heappush(open_list, nuevo_nodo)

    # Si salimos del bucle sin retornar, no hay ruta
    return {
        "encontrada": False,
        "mensaje":    f"No se encontró ruta entre '{origin}' y '{destination}'."
    }


# ---------------------------------------------------------------------------
# Formateo del resultado para presentación al usuario
# ---------------------------------------------------------------------------

def _format_route_message(pasos: list[dict], tiempo_total: float) -> str:
    """
    Convierte la lista de pasos en un mensaje de instrucciones legible.
    """
    if not pasos:
        return "Ruta directa sin pasos intermedios."

    lineas = []
    linea_actual = pasos[0]["linea"]
    inicio_segmento = pasos[0]["desde"]

    for paso in pasos:
        if paso["linea"] != linea_actual or paso["es_transbordo"]:
            lineas.append(f"  🚇 Línea {linea_actual}: {inicio_segmento} → {paso['desde']}")
            if paso["es_transbordo"]:
                lineas.append(f"  🔄 Transbordo en {paso['desde']} → Línea {paso['linea']}")
            linea_actual = paso["linea"]
            inicio_segmento = paso["desde"]

    # Último segmento
    lineas.append(f"  🚇 Línea {linea_actual}: {inicio_segmento} → {pasos[-1]['hasta']}")
    lineas.append(f"\n  ⏱  Tiempo total estimado: {tiempo_total} minutos")

    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Bloque de prueba rápida
# Ejecuta: python route_planner.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    G = build_graph()

    print("\n=== Prueba del planificador de rutas ===\n")

    casos = [
        ("Norte",     "Oriente",      False),
        ("Norte",     "Terminal Sur", True),
        ("Occidente", "Estadio",      False),
    ]

    for origen, destino, pico in casos:
        print(f"Buscando: {origen} → {destino} {'[HORA PICO]' if pico else ''}")
        resultado = find_best_route(origen, destino, hora_pico=pico, G=G)

        if resultado["encontrada"]:
            print(f"  Ruta      : {' → '.join(resultado['ruta'])}")
            print(f"  Tiempo    : {resultado['tiempo_total']} min")
            print(f"  Transbordos: {resultado['num_transbordos']}")
            print(f"  Reglas    : {sorted(resultado['reglas_usadas'])}")
            print(f"  Detalle:\n{resultado['mensaje']}")
        else:
            print(f"  {resultado['mensaje']}")
        print()
