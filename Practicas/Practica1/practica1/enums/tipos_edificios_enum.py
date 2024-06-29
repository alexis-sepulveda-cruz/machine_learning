"""
tipos_edificios.py

Este módulo define un enumerado para los diferentes 
tipos de edificios.

Clases:
    TipoEdificio: Un enumerado que representa los 
    diferentes tipos de edificios.

"""

import re


class TipoEdificio():
    """
    Un enumerado que representa los diferentes 
    tipos de edificios.

    Atributos:
        PREPARATORIA: Una preparatoria.
        FACULTAD: Una facultad.
        HOSPITAL: Un hospital.
        CENTRO: Un centro comunitario.
        ADMIN: Un edificio administrativo.
        OTRO: Cualquier otro tipo de edificio.
    """
    PREPARATORIA = "PREPARATORIA"
    FACULTAD = "FACULTAD"
    HOSPITAL = "HOSPITAL"
    CENTRO = "CENTRO"
    ADMIN = "ADMIN"
    OTRO = "OTRO"

    @classmethod
    def obtener_atributos(cls):
        return {
            nombre: valor \
                for nombre, valor in cls.__dict__.items() \
                    if not nombre.startswith('__') and not callable(valor) and nombre != 'obtener_atributos'
        }


    def get_tipo_edificio(name: str) -> str:
        """
        Categoriza un nombre de edificio en una de las categorías predefinidas.

        Parámetros:
            name (str): El nombre del edificio a categorizar.

        Retorna:
            str: La categoría del edificio, que puede ser una de las siguientes:
                - 'PREPARATORIA': Una preparatoria.
                - 'FACULTAD': Una facultad.
                - 'HOSPITAL': Un hospital.
                - 'CENTRO': Un centro comunitario.
                - 'ADMIN': Un edificio administrativo.
                - 'OTRO': Cualquier otro tipo de edificio.

        Notas:
            - La comparación se realiza sin distinción entre mayúsculas y minúsculas.
            - Si el nombre del edificio no coincide con ninguna categoría predefinida, se asigna a la categoría 'OTRO'.

        Ejemplo:
            >>> categorize('Preparatoria 1')
            'PREPARATORIA'
            >>> categorize('Facultad de Ingeniería')
            'FACULTAD'
            >>> categorize('Hospital General')
            'HOSPITAL'
            >>> categorize('Centro de Investigación')
            'CENTRO'
            >>> categorize('Secretaría Académica')
            'ADMIN'
            >>> categorize('Casa del Estudiante')
            'OTRO'
        """
        name = name.upper()  # Asegurarse de que la comparación no sea sensible a mayúsculas/minúsculas

        patterns = {
            'PREPARATORIA': r'PREPARATORIA|PREPA\.',
            'FACULTAD': r'FACULTAD|FAC\.',
            'HOSPITAL': r'HOSPITAL',
            'CENTRO': r'CENTRO|CTRO\.|C\.|INVESTIGAC',
            'ADMIN': r'SECRETAR[ÍI]A|SRIA\.|DIRECCI[ÓO]N|DEPARTAMENTO|DEPTO\.|CONTRALORIA|AUDITORIA|TESORERIA|'
                    r'ESCOLAR|ABOGAC[ÍI]A|JUNTA|RECTORIA|IMAGEN'
        }

        for category, pattern in patterns.items():
            if re.search(pattern, name):
                return category

        return 'OTRO'