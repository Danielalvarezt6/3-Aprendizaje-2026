"""
    Funciones y clases para entrtenamiento y predicción con árboles de desición numéricos utilizando el criterio de entropía
    
    Se asume que los datos vienen en forma de una lista de diccionarios, donde cada diccionario representa una instancia (la cual puede tener nombres de atributos diferentes)
    
    Todos los atributos son numéricos salvo la clase, la cual puede ser categorica o numerica con pocos valores diferentes.
      
"""

__author__ = "Daniel Alvarez"
__date__ = "Febrero 2026"


import math
import random
from collections import Counter

def entrena_arbol(datos, target, clase_default, 
                  max_profundidad=None, acc_nodo=1.0, min_ejemplos=0,
                  variables_seleccionadas=None):
    """
    Entrena un árbol de desición utilizando el criterio de entropía
    
    Parámetros: 
    -----------
    datos: list(dict)
        Una lista de diccionarios donde cada diccionario representa una instancia. 
        Cada diccionario tiene al menos un par llave-valor, donde la llave es el nombre de un atributo y el valor es el valor del atributo.
        Todos los diccionarios tienen la misma llave-valor. 
    target: str
        El nombre del atributo que se quiere predecir
    clase_default: str
        El valor de la clase por default
    max_profundidad: int
        La máxima profundidad del árbol. Si es None, no hay límite de profundidad
    acc_nodo: int
        El porcentaje de acierto mínimo para considerar un nodo como hoja
    min_ejemplos: int
        El número mínimo de ejemplos para considerar un nodo como hoja
    variables_seleccionadas: list(str)
        Lista de variables a considerar. Si es None, se consideran todas las variables, esto apLica para árboles aleatorios y lo tendrán que implementar en la tarea.
        
    Regresa:
    --------
    nodo: Nodo
        El nodo raíz del árbol de desición
    
    """
    # Criterio para datos vacíos (p. ej. rama sin ejemplos tras un corte)
    if len(datos) == 0:
        return NodoN(terminal=True, clase_default=clase_default)

    atributos = list(datos[0].keys())
    atributos.remove(target)
        
    # Criterios para deterinar si es un nodo hoja
    if len(atributos) == 0:
        return NodoN(terminal=True, clase_default=clase_default)
    
    clases = Counter(d[target] for d in datos)
    clase_default = clases.most_common(1)[0][0]
    
    if (max_profundidad == 0 or 
        len(datos) <= min_ejemplos or 
        clases.most_common(1)[0][1] / len(datos) >= acc_nodo):
        
        return NodoN(terminal=True, clase_default=clase_default)
    
    # Si se especifica `variables_seleccionadas` podemos recibir:
    #  None: usar todos los atributos
    #  una lista de nombres: filtrar los atributos a esa lista
    #  un entero n: elegir aleatoriamente n atributos SOLO para este nodo
    if variables_seleccionadas is not None:
        if isinstance(variables_seleccionadas, int):
            n = max(0, min(variables_seleccionadas, len(atributos)))
            if n == 0:
                return NodoN(terminal=True, clase_default=clase_default)
            atributos = random.sample(atributos, n)
        else:
            atributos = [a for a in atributos if a in variables_seleccionadas]
            if len(atributos) == 0:
                return NodoN(terminal=True, clase_default=clase_default)
    
    variable, valor = selecciona_variable_valor(datos, target, atributos)
    nodo = NodoN(
        terminal=False, 
        clase_default=clase_default,
        atributo=variable, 
        valor=valor 
    )
    
    # Para los hijos: si es entero se pasa el mismo entero (selección aleatoria por nodo)
    # Si es lista, se mantiene la misma lista para filtrar
    # Si es None, se mantiene None
    nodo.hijo_menor = entrena_arbol(
        [d for d in datos if d[variable] < valor],
        target,
        clase_default,
        max_profundidad - 1 if max_profundidad is not None else None,
        acc_nodo, min_ejemplos, variables_seleccionadas
    )   
    nodo.hijo_mayor = entrena_arbol(
        [d for d in datos if d[variable] >= valor],
        target,
        clase_default,
        max_profundidad - 1 if max_profundidad is not None else None,
        acc_nodo, min_ejemplos, variables_seleccionadas
    )   
    return nodo

def selecciona_variable_valor(datos, target, atributos):
    """
    Selecciona el atributo y el valor que mejor separa las clases
    
    Parámetros:
    -----------
    datos: list(dict)
        Una lista de diccionarios donde cada diccionario representa una instancia.
        Cada diccionario tiene al menos un par llave-valor, donde la llave es el nombre de un atributo y el valor es el valor del atributo. Todos los diccionarios tienen la misma llave-valor.
    target: str
        El nombre del atributo que se quiere predecir
    atributos: list(str)
        La lista de atributos a considerar
        
    Regresa:
    --------
    atributo: str
        El nombre del atributo que mejor separa las clases
    valor: float
        El valor del atributo que mejor separa las clases
    """
    
    entropia = entropia_clase(datos, target)
    mejor = max(
        ((a, maxima_ganancia_informacion(datos, target, a, entropia))
            for a in atributos),
        key=lambda x: x[1][1]
    )
    return mejor[0], mejor[1][0]
    
def entropia_clase(datos, target):
    """
    Calcula la entropía de la clase
    
    Parámetros:
    -----------
    datos: list(dict)
        Una lista de diccionarios donde cada diccionario representa una instancia. 
        Cada diccionario tiene al menos un par llave-valor, donde la llave es el nombre de un atributo y el valor es el valor del atributo. Todos los diccionarios tienen la misma llave-valor. 
    target: str
        El nombre del atributo que se quiere predecir
        
    Regresa:
    --------
    entropia: float
        La entropía de la clase
    """ 
      
    clases = Counter(d[target] for d in datos)
    total = sum(clases.values())
    return -sum((c/total) * math.log2(c/total) for c in clases.values())

def maxima_ganancia_informacion(datos, target, atributo, entropia):
    """
    Calcula la ganancia de información de un atributo 

    Parámetros:
    -----------
    datos: list(dict)
        Una lista de diccionarios donde cada diccionario representa una instancia. 
        Cada diccionario tiene al menos un par llave-valor, donde la llave es el nombre de un atributo y el valor es el valor del atributo. Todos los diccionarios tienen la misma llave-valor. 
    target: str
        El nombre del atributo que se quiere predecir
    atributo: str
        El nombre del atributo a considerar
    entropia: float
        La entropía de la clase
        
    Regresa:
    --------
    valor: float
        El valor del atributo que mejor separa las clases
    ganancia: float
        La ganancia de información del atributo dividiendo en ese valor
    
    """
    
    lista_valores = [(d[atributo], d[target]) for d in datos]
    lista_valores.sort(key=lambda x: x[0])
    lista_valor_ganancia = []
    for (v1, v2) in zip(lista_valores[:-1], lista_valores[1:]):
        if v1[1] != v2[1]:
            valor = (v1[0] + v2[0]) / 2
            ganancia = ganancia_informacion(datos, target, atributo, valor, entropia)
            lista_valor_ganancia.append((valor, ganancia))
    return max(lista_valor_ganancia, key=lambda x: x[1])

def ganancia_informacion(datos, target, atributo, valor, entropia):
    """
    Calcula la ganancia de información de un atributo dividiendo en un valor
    
    Parámetros:
    -----------
    datos: list(dict)
        Una lista de diccionarios donde cada diccionario representa una instancia.
        Cada diccionario tiene al menos un par llave-valor, donde la llave es el nombre de un atributo y el valor es el valor del atributo. Todos los diccionarios tienen la misma llave-valor.
    target: str
        El nombre del atributo que se quiere predecir
    atributo: str
        El nombre del atributo a considerar
    valor: float
        El valor del atributo a considerar
    entropia: float
        La entropía de la clase
        
    Regresa:
    --------
    ganancia: float
        La ganancia de información del atributo dividiendo en ese valor
    """
    
    datos_menor = [d for d in datos if d[atributo] < valor]
    datos_mayor = [d for d in datos if d[atributo] >= valor]
    
    entropia_menor = entropia_clase(datos_menor, target)
    entropia_mayor = entropia_clase(datos_mayor, target)
    
    total = len(datos)
    total_menor = len(datos_menor)
    total_mayor = len(datos_mayor)
    
    return (
        entropia 
        - (total_menor / total) * entropia_menor 
        - (total_mayor / total) * entropia_mayor 
    )             

def predice_arbol(arbol, datos):
    return [arbol.predice(d) for d in datos]

def evalua_arbol(arbol, datos, target):
    predicciones = predice_arbol(arbol, datos)
    return sum(1 for p, d in zip(predicciones, datos) if p == d[target]) / len(datos)

def imprime_arbol(nodo, nivel=0):
    if nodo.terminal:
        print("    " * nivel + f"La clase es {nodo.clase_default}")
    else:
        print("    " * nivel + f"Si {nodo.atributo} < {nodo.valor} entonces:")
        imprime_arbol(nodo.hijo_menor, nivel + 1)
        print("    " * nivel + f"Si {nodo.atributo} >= {nodo.valor} entonces:")
        imprime_arbol(nodo.hijo_mayor, nivel + 1)
 
class NodoN:
    def __init__(self, terminal, clase_default, atributo=None, valor=None):
        self.terminal = terminal
        self.clase_default = clase_default
        self.atributo = atributo
        self.valor = valor
        self.hijo_menor = None
        self.hijo_mayor = None
    
    def predice(self, instancia):
        if self.terminal:
            return self.clase_default               
        if instancia[self.atributo] < self.valor:
            return self.hijo_menor.predice(instancia)       
        return self.hijo_mayor.predice(instancia)
    
   
def main():
    datos = [
        {"atributo1": 1, "atributo2": 1, "clase": "positiva"},
        {"atributo1": 2, "atributo2": 1, "clase": "positiva"},
        {"atributo1": 3, "atributo2": 1, "clase": "positiva"},
        {"atributo1": 4, "atributo2": 1, "clase": "positiva"},
        {"atributo1": 1, "atributo2": 2, "clase": "positiva"},
        {"atributo1": 2, "atributo2": 2, "clase": "positiva"},
        {"atributo1": 3, "atributo2": 2, "clase": "positiva"},
        {"atributo1": 4, "atributo2": 2, "clase": "positiva"},
        {"atributo1": 1, "atributo2": 3, "clase": "negativa"},
        {"atributo1": 2, "atributo2": 3, "clase": "negativa"},
        {"atributo1": 3, "atributo2": 3, "clase": "negativa"},
        {"atributo1": 4, "atributo2": 3, "clase": "negativa"},
        {"atributo1": 1, "atributo2": 4, "clase": "positiva"},
        {"atributo1": 2, "atributo2": 4, "clase": "positiva"},
        {"atributo1": 3, "atributo2": 4, "clase": "positiva"},
        {"atributo1": 4, "atributo2": 4, "clase": "positiva"},
     
   ]
    
    raiz = entrena_arbol(datos, "clase", "positiva")
    imprime_arbol(raiz)
    
    acc = evalua_arbol(raiz, datos, "clase")
    print(f"El acierto en los mismos datos que se entrenó es {acc}")
    return None

if __name__ == "__main__":
    main()
