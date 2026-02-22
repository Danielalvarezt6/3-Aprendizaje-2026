"""
Funciones para entrenamiento y predicción con Bosques Aleatorios (Random Forest).
Utiliza como base los árboles de decisión numéricos definidos en arboles_numericos.py.
"""

__author__ = "Daniel Alvarez"
__date__ = "Febrero 2026"

import random
from collections import Counter
import arboles_numericos as an

def entrena_bosque(datos, target, clase_default, n_arboles=10, 
                   max_profundidad=None, acc_nodo=1.0, min_ejemplos=0,
                   variables_seleccionadas=None):
    """
    Entrena un bosque aleatorio generando múltiples árboles de decisión.
    
    Parámetros:
    -----------
    datos: list(dict)
        Lista de diccionarios con las instancias de entrenamiento.
    target: str
        El nombre del atributo que se quiere predecir.
    clase_default: str
        La clase por defecto en caso de empate o nodos vacíos.
    n_arboles: int
        Cantidad de árboles que conformarán el bosque (M subconjuntos).
    max_profundidad: int
        Profundidad máxima permitida para cada árbol.
    acc_nodo: float
        Porcentaje de pureza mínimo para detener la división de un nodo.
    min_ejemplos: int
        Mínimo de ejemplos requeridos en un nodo para continuar dividiendo.
    variables_seleccionadas: int o None
        Número de atributos a seleccionar aleatoriamente en cada nodo del árbol.
        
    Regresa:
    --------
    bosque: list
        Una lista que contiene los nodos raíz de todos los árboles entrenados.
    """
    bosque = []
    n_datos = len(datos)
    
    for _ in range(n_arboles):
        # 1. Separar datos: Muestreo aleatorio con repetición (Bootstrap)
        muestra_bootstrap = random.choices(datos, k=n_datos)
        
        # 2. Entrenar el árbol con el subconjunto y límite de variables
        arbol = an.entrena_arbol(
            datos=muestra_bootstrap,
            target=target,
            clase_default=clase_default,
            max_profundidad=max_profundidad,
            acc_nodo=acc_nodo,
            min_ejemplos=min_ejemplos,
            variables_seleccionadas=variables_seleccionadas
        )
        bosque.append(arbol)
        
    return bosque

def predice_bosque(bosque, datos):
    """
    Realiza predicciones utilizando un bosque aleatorio mediante votación mayoritaria.
    
    Parámetros:
    -----------
    bosque: list
        Lista de árboles (nodos raíz) previamente entrenados.
    datos: list(dict)
        Lista de diccionarios con las instancias a predecir.
        
    Regresa:
    --------
    predicciones_finales: list
        Lista con las clases predichas para cada instancia.
    """
    predicciones_finales = []
    
    for instancia in datos:
        # Obtener la predicción individual de cada árbol en el bosque
        votos = [arbol.predice(instancia) for arbol in bosque]
        
        # 3. Votación mayoritaria: gana la clase más común
        conteo = Counter(votos)
        clase_ganadora = conteo.most_common(1)[0][0]
        predicciones_finales.append(clase_ganadora)
        
    return predicciones_finales

def evalua_bosque(bosque, datos, target):
    """
    Evalúa la exactitud (accuracy) del bosque aleatorio en un conjunto de datos.
    """
    predicciones = predice_bosque(bosque, datos)
    aciertos = sum(1 for p, d in zip(predicciones, datos) if p == d[target])
    return aciertos / len(datos)