import os
import random
from collections import Counter
import utileria as ut
import bosque_aleatorio as ba

__author__ = "Daniel Alvarez"
__date__ = "Febrero 2026"

# Configuración de los datos (Iris dataset)
url = 'https://archive.ics.uci.edu/static/public/53/iris.zip'
archivo = 'datos/iris.zip'
archivo_datos = 'datos/iris.data'
atributos = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
target = 'class'

# Descarga y descomprime los datos utilizando utileria.py
if not os.path.exists('datos'):
    os.makedirs('datos')
if not os.path.exists(archivo):
    ut.descarga_datos(url, archivo)
    ut.descomprime_zip(archivo, directorio='datos')

# Lee los datos
datos_crudos = ut.lee_csv(archivo_datos, atributos=atributos, separador=',')

# Limpieza y conversión a valores numéricos
datos = []
for d in datos_crudos:
    # Se omiten líneas en blanco que puedan venir al final del archivo
    if d['sepal_length'] == '' or d['class'] == '':
        continue
    try:
        datos.append({
            'sepal_length': float(d['sepal_length']),
            'sepal_width': float(d['sepal_width']),
            'petal_length': float(d['petal_length']),
            'petal_width': float(d['petal_width']),
            'class': d['class']
        })
    except ValueError:
        continue

# Determina la clase por defecto (la más común)
clases = Counter(d[target] for d in datos)
clase_default = clases.most_common(1)[0][0]

# Selecciona un conjunto de entrenamiento y de validación
random.seed(42)
random.shuffle(datos)
N = int(0.8 * len(datos))
datos_entrenamiento = datos[:N]
datos_validacion = datos[N:]

# Test 1: Efecto del número de árboles
print("Test 1: Variando el número de árboles (max_prof=3, var_sel=None)")
resultados_n = []
for n in [1, 3, 5, 10, 20]:
    # Entrena el bosque
    bosque = ba.entrena_bosque(
        datos_entrenamiento, target, clase_default, 
        n_arboles=n, max_profundidad=3
    )
    # Evalúa el bosque
    acc_in = ba.evalua_bosque(bosque, datos_entrenamiento, target)
    acc_out = ba.evalua_bosque(bosque, datos_validacion, target)
    resultados_n.append((n, acc_in, acc_out))

print('N_arboles'.center(15) + 'Acc_in'.center(15) + 'Acc_out'.center(15))
print('-' * 45)
for n, acc_in, acc_out in resultados_n:
    print(f'{n}'.center(15) + f'{acc_in:.4f}'.center(15) + f'{acc_out:.4f}'.center(15))
print('-' * 45 + '\n')


# Test 2: Efecto de la profundidad máxima
print("Test 2: Variando profundidad máxima (n_arboles=10, var_sel=None)")
resultados_prof = []
for profundidad in [1, 2, 3, 4, None]:
    bosque = ba.entrena_bosque(
        datos_entrenamiento, target, clase_default, 
        n_arboles=10, max_profundidad=profundidad
    )
    acc_in = ba.evalua_bosque(bosque, datos_entrenamiento, target)
    acc_out = ba.evalua_bosque(bosque, datos_validacion, target)
    resultados_prof.append((profundidad, acc_in, acc_out))

print('Profundidad'.center(15) + 'Acc_in'.center(15) + 'Acc_out'.center(15))
print('-' * 45)
for prof, acc_in, acc_out in resultados_prof:
    p_str = str(prof) if prof is not None else "None"
    print(f'{p_str}'.center(15) + f'{acc_in:.4f}'.center(15) + f'{acc_out:.4f}'.center(15))
print('-' * 45 + '\n')


# Test 3: Efecto de las variables seleccionadas por nodo
print("Test 3: Variando variables por nodo (n_arboles=10, max_prof=3)")
resultados_var = []
for var in [1, 2, 3, 4]:
    bosque = ba.entrena_bosque(
        datos_entrenamiento, target, clase_default, 
        n_arboles=10, max_profundidad=3, variables_seleccionadas=var
    )
    acc_in = ba.evalua_bosque(bosque, datos_entrenamiento, target)
    acc_out = ba.evalua_bosque(bosque, datos_validacion, target)
    resultados_var.append((var, acc_in, acc_out))

print('Variables'.center(15) + 'Acc_in'.center(15) + 'Acc_out'.center(15))
print('-' * 45)
for var, acc_in, acc_out in resultados_var:
    print(f'{var}'.center(15) + f'{acc_in:.4f}'.center(15) + f'{acc_out:.4f}'.center(15))
print('-' * 45 + '\n')


# Y Ahora entrenamos un bosque por única vez con todos los datos
print("Entrenando un bosque final con todos los datos (n_arboles=15, max_profundidad=4)")
bosque_final = ba.entrena_bosque(
    datos, target, clase_default, 
    n_arboles=15, max_profundidad=4
)

acc_final = ba.evalua_bosque(bosque_final, datos, target)
print(f"Exactitud del bosque final sobre el conjunto de datos completo: {acc_final:.4f}")

#CONCLUSIONES:
#Con el dataset Iris, se puede observar que pocos arboles son suficientes para obtener un buen resultado,
#la profundidad tiene un efecto significativo, se puede observar que a medida que se aumenta la profundidad, el resultado mejora ligeramente.
#Al llegar a profundidad 4, se estabiliza y sin limite de profundidad, el a exactitud en entrenamiento es muy alta (≈99.2 %) 
#pero en validación es algo menor (≈96.7 %), lo que apunta a un posible sobreajuste cuando los árboles son demasiado profundos.
#En cuanto a las variables seleccionadas por nodo, se puede observar que no tiene un efecto significativo, aunque a medida que se aumenta el numero de variables, el resultado mejora ligeramente.
#El bosque final entrenado con todos los datos (15 árboles, profundidad 4) obtiene una exactitud de 99.33 % sobre el conjunto completo.