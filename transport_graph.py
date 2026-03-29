"""
transport_graph.py
------------------
Archivo que representa el sistema de transporte como un grafo.

Un grafo es una estructura que tiene:
    - Nodos    → en este caso son las estaciones
    - Aristas  → las conexiones entre estaciones (los tramos)
    - Pesos    → el tiempo en minutos que tarda cada tramo

Para construir el grafo usé la librería networkx, que ya tiene
todas las funciones necesarias para crear y recorrer grafos
sin tener que programarlas desde cero.

Decidí usar un grafo dirigido porque en algunos sistemas de
transporte hay tramos que solo funcionan en una dirección.
Aunque en este proyecto los tramos son de doble vía, usar
un grafo dirigido me da más flexibilidad si en el futuro
necesito agregar tramos de una sola dirección.
"""

import networkx as nx
from knowledge_base import LINES, STATIONS, get_station_info, get_line_info


# ---------------------------------------------------------------------------
# Definición de aristas del grafo
# Formato: (estacion_a, estacion_b, linea, tiempo_nominal_min)
#
# El tiempo nominal viene de LINES["X"]["tiempo_base_min"].
# Lo multiplicamos por la cantidad de "paradas" entre las dos estaciones,
# pero aquí lo simplificamos: cada par adyacente = 1 parada = tiempo_base.
# ---------------------------------------------------------------------------

EDGES: list[tuple[str, str, str]] = [
    # Línea A: Norte ↔ Centro ↔ Portal del Quindio  ↔ Sur
    ("Norte",       "Centro",     "A"),
    ("Centro",     "Portal del Quindio", "A"),
    ("Portal del Quindio", "Sur",         "A"),

    # Línea B: El Bosque ↔ Centro ↔ Mercado Minorista Quindiano ↔ La Clarita
    ("El Bosque",   "Centro",     "B"),
    ("Centro",     "Mercado Minorista Quindiano",     "B"),
    ("Mercado Minorista Quindiano",     "La Clarita",     "B"),

    # Línea C: Portal del Quindio ↔ UUniversidad Del Quindio  ↔ Estadio Centenario ↔ Mercado Minorista Quindiano ↔ Terminal del Sur
    ("Portal del Quindio", "Universidad Del Quindio", "C"),
    ("Universidad Del Quindio", "Estadio Centenario",     "C"),
    ("Estadio Centenario",     "Mercado Minorista Quindiano",     "C"),
    ("Mercado Minorista Quindiano",     "Terminal del Sur","C"),
]


def build_graph() -> nx.DiGraph:
    """
    Construye y retorna el grafo dirigido del sistema de transporte.

    Para cada arista definida en EDGES:
      1. Calcula el tiempo base desde LINES
      2. Agrega la arista en ambas direcciones (bidireccional)
      3. Guarda metadatos: línea, tiempo, estaciones origen/destino

    Retorna:
        nx.DiGraph: grafo listo para ser consultado por el planificador.
    """
    G = nx.DiGraph()

    # Agregar nodos con sus atributos (información de la estación)
    for station_name, attrs in STATIONS.items():
        G.add_node(station_name, **attrs)

    # Agregar aristas con sus pesos
    for origin, destination, line in EDGES:
        line_data = get_line_info(line)
        tiempo_base = line_data.get("tiempo_base_min", 5)

        # Atributos que viajan con cada arista
        edge_attrs = {
            "linea":    line,
            "weight":   tiempo_base,   # 'weight' es la clave estándar de networkx
            "tiempo":   tiempo_base,
        }

        # Agregar en ambas direcciones
        G.add_edge(origin, destination, **edge_attrs)
        G.add_edge(destination, origin, **edge_attrs)

    return G


def get_neighbors(G: nx.DiGraph, station: str) -> list[dict]:
    """
    Retorna los vecinos de una estación con los datos de la arista.

    Ejemplo de retorno:
        [
          {"estacion": "Centro", "linea": "A", "tiempo": 4},
          {"estacion": "Sur",     "linea": "A", "tiempo": 4},
        ]

    Args:
        G:       el grafo construido con build_graph()
        station: nombre de la estación actual
    """
    neighbors = []
    for neighbor in G.successors(station):
        edge_data = G.get_edge_data(station, neighbor)
        neighbors.append({
            "estacion": neighbor,
            "linea":    edge_data.get("linea", "?"),
            "tiempo":   edge_data.get("tiempo", 5),
        })
    return neighbors


def print_graph_summary(G: nx.DiGraph) -> None:
    """
    Imprime un resumen del grafo. Útil para depuración.
    """
    print(f"\n{'='*50}")
    print(f"  RESUMEN DEL GRAFO DE TRANSPORTE")
    print(f"{'='*50}")
    print(f"  Estaciones (nodos) : {G.number_of_nodes()}")
    print(f"  Tramos    (aristas): {G.number_of_edges()}")
    print(f"\n  Conexiones por estación:")
    for node in sorted(G.nodes()):
        vecinos = [f"{v}({G[node][v]['linea']})" for v in G.successors(node)]
        print(f"    {node:20s} → {', '.join(vecinos)}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Bloque de prueba rápida
# Ejecuta: python transport_graph.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    grafo = build_graph()
    print_graph_summary(grafo)

    # Verificación puntual
    vecinos_central = get_neighbors(grafo, "Centro")
    print("Vecinos de Centro:")
    for v in vecinos_central:
        print(f"  → {v['estacion']:20s} | Línea {v['linea']} | {v['tiempo']} min")
