"""
columnas_enum.py

Este módulo define un enumerado para las columnas de un 
dataset de empleados.

El dataset de empleados contiene información sobre el nombre, 
sueldo neto, dependencia, fecha y tipo de empleado. Este módulo 
define un enumerado llamado `Columna` que representa cada una 
de estas columnas.

Clases:
    Columna: Un enumerado que representa las columnas del dataset 
    de empleados.

Ejemplo:
    Para utilizar el enumerado `Columna`, primero debes importarlo 
    desde este módulo:

    >>> from columnas_enum import Columna

    Luego, puedes acceder a los valores del enumerado utilizando la 
    notación de punto:

    >>> columna = Columna.NOMBRE
    >>> print(columna)
    Columna.NOMBRE

    También puedes comparar los valores del enumerado con cadenas 
    de texto:

    >>> if columna == "Nombre":
    ...     print("La columna es 'Nombre'")
    La columna es 'Nombre'

Notas:
    - El enumerado `Columna` utiliza la clase `Enum` del módulo 
      estándar `enum`.
    - Los nombres de las columnas están definidos en mayúsculas 
      para seguir la convención de nombres de enumerados en Python.
"""
# from enum import Enum


class EmpleadoColumna():
    """
    Un enumerado que representa las columnas del dataset de 
    empleados.

    Atributos:
        NOMBRE (str): La columna "Nombre del empleado".
        SUELDO_NETO (str): La columna "Sueldo Neto del empleado".
        DEPENDENCIA (str): La columna "Dependencia donde labora el empleado".
        FECHA (str): La columna "Fecha en la cual se tranparento el pago al empleado".
        TIPO (str): La columna "Tipo de edificio en el que labora".
    """
    NOMBRE = "Nombre"
    SUELDO_NETO = "Sueldo Neto"
    DEPENDENCIA = "dependencia"
    FECHA = "Fecha"
    ANIO = "Anio"
    TIPO = "Tipo"