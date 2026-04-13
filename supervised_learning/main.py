"""
main.py
-------
Archivo principal del sistema de rutas para transporte masivo.

Este archivo es el que el usuario ejecuta directamente.
Su función es coordinar los demás módulos del proyecto:

    1. Mostrar el menú con las opciones disponibles
    2. Pedir al usuario la estación de origen y destino
    3. Enviar esos datos al planificador de rutas
    4. Mostrar la ruta encontrada de forma clara

Nota: toda la lógica del sistema (reglas, grafo, algoritmo)
está distribuida en los otros archivos del proyecto.
Este archivo solo los llama y muestra los resultados.

Para ejecutar:
    python main.py

Requiere tener instalado:
    pip install networkx
"""

from transport_graph import build_graph, print_graph_summary
from route_planner import find_best_route
from knowledge_base import STATIONS


# ---------------------------------------------------------------------------
# Constantes de presentación
# ---------------------------------------------------------------------------

BANNER = """
╔══════════════════════════════════════════════════════════╗
║     SISTEMA INTELIGENTE DE RUTAS — TRANSPORTE MASIVO     ║
║             EN LA CIUDAD DE ARMENIA QUINDIO              ║
║       Base de conocimiento + Motor de inferencia + A*    ║
╚══════════════════════════════════════════════════════════╝
"""


def mostrar_estaciones() -> None:
    """Imprime la lista de estaciones disponibles con sus líneas."""
    print("\n  Estaciones disponibles:")
    print("  " + "-" * 46)
    for nombre, datos in sorted(STATIONS.items()):
        lineas = ", ".join(datos["lineas"])
        flag   = "🔀" if datos["es_intercambio"] else "  "
        print(f"  {flag}  {nombre:20s}  [Línea(s): {lineas}]")
    print("  " + "-" * 46)
    print("  🔀 = Estación de intercambio (transbordo disponible)\n")


def pedir_estacion(prompt: str) -> str:
    """
    Solicita al usuario que ingrese una estación válida.
    Repite hasta que ingrese una que exista en el sistema.
    """
    while True:
        entrada = input(prompt).strip().title()
        if entrada in STATIONS:
            return entrada
        print(f"  ⚠  '{entrada}' no existe. Revisa la lista de estaciones.\n")


def pedir_hora_pico() -> bool:
    """Pregunta si es hora pico. Retorna True o False."""
    respuesta = input("  ¿Es hora pico? (s/n): ").strip().lower()
    return respuesta in ("s", "si", "sí", "y", "yes")


def mostrar_resultado(resultado: dict) -> None:
    """Formatea y muestra el resultado de la búsqueda de ruta."""
    print("\n" + "═" * 58)

    if not resultado["encontrada"]:
        print(f"  ❌  {resultado['mensaje']}")
        print("═" * 58)
        return

    print("  ✅  RUTA ENCONTRADA")
    print("═" * 58)
    print(f"\n  Estaciones: {' → '.join(resultado['ruta'])}")
    print(f"  Tiempo total : {resultado['tiempo_total']} minutos")
    print(f"  Transbordos  : {resultado['num_transbordos']}")

    if resultado.get("reglas_usadas"):
        print(f"  Reglas usadas: {', '.join(sorted(resultado['reglas_usadas']))}")

    print("\n  📋 Instrucciones paso a paso:")
    print(resultado["mensaje"])

    if resultado["pasos"]:
        print("\n  📊 Desglose de tiempos:")
        for paso in resultado["pasos"]:
            icono = "🔄" if paso["es_transbordo"] else "🚇"
            print(f"    {icono}  {paso['desde']:18s} → {paso['hasta']:18s} "
                  f"[Línea {paso['linea']}]  {paso['tiempo']:5.1f} min")

    print("\n" + "═" * 58)


def menu_principal(G) -> None:
    """
    Bucle principal del menú de la aplicación.
    """
    while True:
        print("\n  ¿Qué deseas hacer?")
        print("  [1] Buscar ruta entre dos estaciones")
        print("  [2] Ver mapa del sistema (grafo)")
        print("  [3] Salir")

        opcion = input("\n  Tu opción: ").strip()

        if opcion == "1":
            print()
            mostrar_estaciones()
            origen  = pedir_estacion("  Estación de ORIGEN  : ")
            destino = pedir_estacion("  Estación de DESTINO : ")
            pico    = pedir_hora_pico()

            print("\n  🔍 Calculando ruta óptima...")
            resultado = find_best_route(origen, destino, hora_pico=pico, G=G)
            mostrar_resultado(resultado)

        elif opcion == "2":
            print_graph_summary(G)

        elif opcion == "3":
            print("\n  👋 Hasta pronto.\n")
            break

        else:
            print("  ⚠  Opción no válida. Ingresa 1, 2 o 3.")


# ---------------------------------------------------------------------------
# Punto de entrada Python estándar
# ---------------------------------------------------------------------------

def main() -> None:
    print(BANNER)
    print("  Cargando el sistema...")

    # Construimos el grafo UNA SOLA VEZ y lo reutilizamos en toda la sesión
    G = build_graph()

    print("   Grafo de transporte listo.")
    print(f"  Base de conocimiento con reglas activas.")
    print(f"  Motor de inferencia inicializado.\n")

    menu_principal(G)


if __name__ == "__main__":
    main()
