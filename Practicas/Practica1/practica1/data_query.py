import pandas as pd

from typing import Any
from practica1.enums.tipos_edificios_enum import TipoEdificio
from practica1.enums.empleado_columnas_enum import EmpleadoColumna as Columna
from practica1.utils.constantes_util import ConstantesUtil
from practica1.utils.util import get_data
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns


class DataQuery():
    def __init__(self):
        self.df_typed = get_data()
        self.df_updated = get_data('update_uanl.csv')

    def std_mean_annual_by_dependencia(self, start_year: int = 2019, use_update_data: bool = True) -> pd.DataFrame:

        if use_update_data:
            df_work = self.df_updated
        else:
            df_work = self.df_typed

        # Filtrar df para obtener solo las filas con ANIO == start_year
        df_filtrado = df_work.query(f"{Columna.ANIO} == {start_year}")

        # Obtener todas las dependencias únicas a partir de start_year
        dependencias = df_filtrado[Columna.DEPENDENCIA].unique()

        # Filtrar df para obtener solo las filas con las dependencias deseadas
        df_filtrado = df_work[df_work[Columna.DEPENDENCIA].isin(dependencias)].reset_index()

        # Agrupar las dependencias y el año
        grouped_dep = df_filtrado \
            .groupby([Columna.DEPENDENCIA, Columna.ANIO]) \
                [Columna.SUELDO_NETO].agg(ConstantesUtil.ESTADISTICA_DESCRIPTIVA) \
                    .reset_index()

        # Medir la variabilidad promedio por dependencia
        variability = grouped_dep \
            .groupby(Columna.DEPENDENCIA)['std'].mean().reset_index() \
            .rename(columns={'std': 'std_mean'}) \
            .sort_values(by='std_mean')
        
        return variability
    
    def std_mean_annual_by_tipo_edificio(self, start_year: int = 2019, use_update_data: bool = True) -> pd.DataFrame:
        if use_update_data:
            df_work = self.df_updated
        else:
            df_work = self.df_typed

        # Filtrar df para obtener solo las filas con ANIO == start_year
        df_filtrado = df_work.query(f"{Columna.ANIO} == {start_year}")

        # Obtener todas las dependencias únicas a partir de start_year
        dependencias = df_filtrado[Columna.DEPENDENCIA].unique()

        # Filtrar df para obtener solo las filas con las dependencias deseadas
        df_filtrado = df_work[df_work[Columna.DEPENDENCIA].isin(dependencias)].reset_index()

        # Agrupar las dependencias y el año
        grouped_dep = df_filtrado \
            .groupby([Columna.TIPO, Columna.DEPENDENCIA, Columna.ANIO]) \
                [Columna.SUELDO_NETO].agg(ConstantesUtil.ESTADISTICA_DESCRIPTIVA) \
                    .reset_index()

        # Medir la variabilidad promedio por dependencia
        variability = grouped_dep \
            .groupby([Columna.TIPO, Columna.DEPENDENCIA])['std'].mean().reset_index() \
            .rename(columns={'std': 'std_mean'}) \
            .sort_values(by='std_mean')
        
        return variability
    
    def std_mean_analysis(self, start_year: int = 2019) -> pd.DataFrame:
        def _get_menor_mayor(variabilidad: pd.DataFrame, tipo: str) -> pd.DataFrame:
            # Asignar índices del 1 al 10 para menor y mayor variabilidad
            menor_variabilidad = variabilidad.head(10).reset_index().assign(Tipo=tipo, mayor_menor='MENOR')
            menor_variabilidad['index'] = range(1, len(menor_variabilidad) + 1)

            mayor_variabilidad = variabilidad.tail(10).sort_values('std_mean', ascending=False).reset_index().assign(Tipo=tipo, mayor_menor='MAYOR')
            mayor_variabilidad['index'] = range(1, len(mayor_variabilidad) + 1)
            
            return pd.concat([menor_variabilidad, mayor_variabilidad])

        std_mean_global = self.std_mean_annual_by_dependencia(start_year)
        variabilidad_tipo_edificio = self.std_mean_annual_by_tipo_edificio(start_year)

        # Procesar resultado global
        resultados = []
        resultados.append(_get_menor_mayor(std_mean_global, 'GLOBAL'))

        # Procesar resultados por tipo
        atributos = TipoEdificio.obtener_atributos()
        for _, valor in atributos.items():
            df_filtrado = variabilidad_tipo_edificio.query(f"{Columna.TIPO} == '{valor}'")
            resultados.append(_get_menor_mayor(df_filtrado, valor))

        # Concatenar todos los resultados en un único DataFrame
        df_resultado = pd.concat(resultados).reset_index(drop=True)

        reordenar_columnas = ['index', 'mayor_menor', 'Tipo', 'dependencia', 'std_mean']

        return df_resultado[reordenar_columnas]
    
    def std_mean_descriptivo(self) -> pd.DataFrame:
        # Obtener el análisis de la media y desviación estándar
        std_mean = self.std_mean_analysis()

        # Filtrar las dependencias que no son globales
        dependencias_no_globales = std_mean[std_mean['Tipo'] != "GLOBAL"][Columna.DEPENDENCIA]

        # Filtrar el DataFrame actualizado por dependencias no globales
        std_mean_filter = self.df_updated[self.df_updated[Columna.DEPENDENCIA].isin(dependencias_no_globales)]

        # Fusionar el DataFrame filtrado con std_mean para obtener mayor_menor y dependencia
        df_mayor_menor = std_mean_filter.merge(
            std_mean[['mayor_menor', Columna.DEPENDENCIA]].drop_duplicates(subset=Columna.DEPENDENCIA),
            on=Columna.DEPENDENCIA,
            how='left'
        )

        #df_mayor_menor.groupby([Columna.DEPENDENCIA, Columna.ANIO])

        # Histogramas
        plt.figure(figsize=(10, 6))
        sns.histplot(df_mayor_menor[Columna.SUELDO_NETO], bins=30, kde=True)
        plt.title('Distribución de Sueldos Netos')
        plt.xlabel(Columna.SUELDO_NETO)
        plt.ylabel('Frecuencia')
        plt.show()
        plt.savefig("alexis_barras.png")
        plt.close()

        # Calcular estadísticas descriptivas agrupadas por dependencia, año y mayor_menor
        df_descriptivo = df_mayor_menor.groupby([Columna.DEPENDENCIA, Columna.ANIO, 'mayor_menor'])[Columna.SUELDO_NETO].agg(['min', 'max', 'mean', 'std']).reset_index()

        # Fusionar std_mean con las estadísticas descriptivas
        std_mean_descriptivo = std_mean.merge(df_descriptivo, on=Columna.DEPENDENCIA, how='inner')

        return std_mean_descriptivo

    def search_new_dependencias(self) -> pd.DataFrame:
        # Obtener dependencias únicas por año
        dependencias_por_anio = self.df_typed.groupby(Columna.ANIO)[Columna.DEPENDENCIA].unique().reset_index()

        # Crear un diccionario de dependencias por año
        dependencias_dict = {row[Columna.ANIO]: set(row[Columna.DEPENDENCIA]) for _, row in dependencias_por_anio.iterrows()}

        # Identificar nuevas dependencias por año
        nuevas_dependencias = {}
        anios = sorted(dependencias_dict.keys())

        for i in range(1, len(anios)):
            anio_actual = anios[i]
            anio_anterior = anios[i - 1]
            nuevas_dependencias[anio_actual] = dependencias_dict[anio_actual] - dependencias_dict[anio_anterior]

        return nuevas_dependencias

    def search_missing_dependencias(self) -> pd.DataFrame:
        # Obtener dependencias únicas por año
        dependencias_por_anio = self.df_typed.groupby(Columna.ANIO)[Columna.DEPENDENCIA].unique().reset_index()

        # Crear un diccionario de dependencias por año
        dependencias_dict = {row[Columna.ANIO]: set(row[Columna.DEPENDENCIA]) for index, row in dependencias_por_anio.iterrows()}

        # Identificar dependencias desaparecidas por año
        dependencias_desaparecidas = {}
        anios = sorted(dependencias_dict.keys(), reverse=False)

        for i in range(len(anios) - 1):
            anio_actual = anios[i]
            anio_siguiente = anios[i + 1]
            desaparecidas = dependencias_dict[anio_actual] - dependencias_dict[anio_siguiente]
            dependencias_desaparecidas[anio_siguiente] = desaparecidas

        return dependencias_desaparecidas
    
    def search_change_name_dependencias(self) -> pd.DataFrame:
        # Obtener dependencias únicas por año
        dependencias_por_anio = self.df_typed.groupby(Columna.ANIO)[Columna.DEPENDENCIA].unique().reset_index()

        # Crear un diccionario de dependencias por año
        dependencias_dict = {row[Columna.ANIO]: set(row[Columna.DEPENDENCIA]) for _, row in dependencias_por_anio.iterrows()}

        # Identificar dependencias desaparecidas por año
        anios = sorted(dependencias_dict.keys())
        dependencias_cambio_nombre = []

        for i in range(len(anios) - 1):
            anio_actual, anio_siguiente = anios[i], anios[i + 1]
            desaparecidas = dependencias_dict[anio_actual] - dependencias_dict[anio_siguiente]

            for dependencia in desaparecidas:
                df_dependencia_actual = self.df_typed[self.df_typed[Columna.DEPENDENCIA] == dependencia].sort_values(Columna.FECHA, ascending=False)
                ultimo_fecha = df_dependencia_actual.iloc[0][Columna.FECHA]
                empleados = df_dependencia_actual[df_dependencia_actual[Columna.FECHA] == ultimo_fecha][Columna.NOMBRE]
                resultado = self.df_typed[(self.df_typed[Columna.FECHA] > ultimo_fecha) & (self.df_typed[Columna.NOMBRE].isin(empleados))]

                if resultado.empty:
                    dependencias_cambio_nombre.append({
                        "Anterior_Dependencia": dependencia,
                        "Nueva_Dependencia": "DESAPARECE",
                        "Fecha_Cambio": ultimo_fecha
                    })
                else:
                    ultima_fecha_resultado = resultado[Columna.FECHA].min()
                    resultado_filtrado = resultado[resultado[Columna.FECHA] == ultima_fecha_resultado]
                    grupo_dependencia = resultado_filtrado[Columna.DEPENDENCIA].value_counts().reset_index()
                    grupo_dependencia.columns = [Columna.DEPENDENCIA, 'conteo']

                    if not grupo_dependencia.empty:
                        mayor_conteo_dependencia = grupo_dependencia.iloc[0][Columna.DEPENDENCIA]
                        dependencias_cambio_nombre.append({
                            "Anterior_Dependencia": dependencia,
                            "Nueva_Dependencia": mayor_conteo_dependencia,
                            "Fecha_Cambio": ultimo_fecha
                        })
                    else:
                        dependencias_cambio_nombre.append({
                            "Anterior_Dependencia": dependencia,
                            "Nueva_Dependencia": "DESAPARECE",
                            "Fecha_Cambio": ultimo_fecha
                        })

        return pd.DataFrame(dependencias_cambio_nombre)
    
    def create_df_change_name_dependencias(self) -> pd.DataFrame:
        dependencias_cambio_nombre = self.search_change_name_dependencias()
        df_original = self.df_typed.copy()

        # Crear un diccionario para la actualización
        cambio_nombre_dict = {
            row["Anterior_Dependencia"]: row["Nueva_Dependencia"]
            for _, row in dependencias_cambio_nombre.iterrows()
        }

        # Función para actualizar los nombres de las dependencias
        def _actualizar_dependencia(dependencia: str):
            return cambio_nombre_dict.get(dependencia, dependencia)

        # Aplicar la función de actualización
        df_original[Columna.DEPENDENCIA] = df_original[Columna.DEPENDENCIA].apply(_actualizar_dependencia)

        # Guardar el resultado en un archivo CSV
        df_original.to_csv("csv/update_uanl.csv", index=False)

        return df_original
    
    def search_missing_dependencias2(self) -> pd.DataFrame:
        # Obtener dependencias únicas por año
        dependencias_por_anio = self.df_typed.groupby(Columna.ANIO)[Columna.DEPENDENCIA].unique().reset_index()

        # Crear un diccionario de dependencias por año
        dependencias_dict = {row[Columna.ANIO]: set(row[Columna.DEPENDENCIA]) for index, row in dependencias_por_anio.iterrows()}

        # Identificar dependencias desaparecidas por año
        anios = sorted(dependencias_dict.keys(), reverse=True)

        # Encontrar dependencias que no existen en ningún año posterior
        dependencias_no_existentes_posteriores = {}
        for i in range(len(anios) - 1):
            anio_actual = anios[i]
            dependencias_no_existentes = dependencias_dict[anio_actual]
            for j in range(i + 1, len(anios)):
                anio_posterior = anios[j]
                dependencias_no_existentes -= dependencias_dict[anio_posterior]
            dependencias_no_existentes_posteriores[anio_actual] = dependencias_no_existentes

        return dependencias_no_existentes_posteriores
    
    def search_common_words_in_dependency(self) -> list[tuple[Any, int]]:
        # Lista de palabras vacías en español
        stop_words = set([
            'de', 'y', 'la', 'del', 'en', 'e', 'las', 'al', 'para', 
            'u', 'a', 'n', 'l', 'el', 'los', 'un', 'una', 'unos', 'unas', 'con',
            'por', 'sin', 'sobre', 'entre', 'hasta', 'desde', 'hasta', 'durante',
            'ante', 'bajo', 'cabe', 'contra', 'hacia', 'hasta', 'mediante',
            'para', 'por', 'según', 'sin', 'so', 'tras', 'versus', 'vía', 'c'
        ])

        # Extraer las dependencias únicas
        dependencias = self.df_typed[Columna.DEPENDENCIA].unique()

        # Tokenizar los nombres de las dependencias en palabras
        words = []
        for dep in dependencias:
            words.extend(re.findall(r'\b\w+\b', dep.lower()))
        
        # Filtrar números y palabras vacías
        filtered_words = [word for word in words if word not in stop_words and not word.isdigit()]
        
        # Contar la frecuencia de cada palabra
        word_count = Counter(filtered_words)
        
        # Mostrar las palabras más comunes
        return word_count.most_common()