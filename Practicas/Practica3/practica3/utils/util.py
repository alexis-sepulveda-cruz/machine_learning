import numbers
import os
import numpy as np
from practica3.utils.column_assets import ColumnAssets as Column
import pandas as pd


asset_groups = {
    'financial_assets': [Column.S_P_500_PRICE, Column.NASDAQ_100_PRICE, Column.BITCOIN_PRICE, Column.ETHEREUM_PRICE],
    'commodities': [Column.NATURAL_GAS_PRICE, Column.CRUDE_OIL_PRICE, Column.COPPER_PRICE, Column.GOLD_PRICE, Column.SILVER_PRICE, Column.PLATINUM_PRICE],
    'tech_stocks': [Column.APPLE_PRICE, Column.MICROSOFT_PRICE, Column.GOOGLE_PRICE, Column.AMAZON_PRICE, Column.META_PRICE, Column.NETFLIX_PRICE, Column.NVIDIA_PRICE, Column.TESLA_PRICE],
    'others': [Column.BERKSHIRE_PRICE]
}

activos = [
    Column.S_P_500_PRICE, Column.NASDAQ_100_PRICE, Column.BITCOIN_PRICE, Column.ETHEREUM_PRICE,
    Column.NATURAL_GAS_PRICE, Column.CRUDE_OIL_PRICE, Column.COPPER_PRICE, Column.GOLD_PRICE, Column.SILVER_PRICE, Column.PLATINUM_PRICE,
    Column.APPLE_PRICE, Column.MICROSOFT_PRICE, Column.GOOGLE_PRICE, Column.AMAZON_PRICE, Column.META_PRICE, Column.NETFLIX_PRICE,
    Column.NVIDIA_PRICE, Column.TESLA_PRICE, Column.BERKSHIRE_PRICE
]

periodos = [7, 30, 90]


def calcular_tendencia(serie: pd.DataFrame, ventana):
    cambio_porcentual = (serie.pct_change(periods=ventana) * 100).fillna(0)
    condiciones = [
        (cambio_porcentual > 2),
        (cambio_porcentual < -2),
        ((cambio_porcentual >= -2) & (cambio_porcentual <= 2))
    ]
    opciones = ['alcista', 'bajista', 'lateral']
    return np.select(condiciones, opciones, default='lateral')

def get_data(data_file: str = 'stock_market_dataset.csv') -> pd.DataFrame:
    # Obtener los nombres de las columnas del archivo CSV
    column_names = pd.read_csv(f'csv/{data_file}', nrows=0).columns

    # Leer el archivo CSV sin incluir la primera columna
    df = pd.read_csv(f'csv/{data_file}', usecols=column_names[1:])

    # Convertir la columna 'Fecha' a tipo datetime
    df[Column.DATE] = pd.to_datetime(df[Column.DATE], format="%d-%m-%Y")

    # Extraer el año de la columna 'Fecha'
    df[Column.YEAR] = df[Column.DATE].dt.year

    # Extraer el mes de la columna 'Fecha'
    df[Column.MONTH] = df[Column.DATE].dt.month

    # Eliminar los separadores de miles de las columnas
    df = df.replace({',': ''}, regex=True)

    # Convertir todas las columnas, excepto 'Date' y 'YEAR', en flotantes
    for column in df.columns.difference([Column.DATE, Column.YEAR, Column.MONTH]):
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            pass

        # if column in activos:
        #     for periodo in periodos:
        #         nueva_columna = f"{column}_tendencia_{periodo}d"
        #         df[nueva_columna] = calcular_tendencia(df[column], periodo)

    # Se ordena el data set por fecha
    df = df.sort_values(Column.DATE)

    return df

def filter_data_by_year(df, year) -> pd.DataFrame:
    """
    Filtra los datos por un año específico.
    """
    # Filtrar los datos por año
    filtered_data = df[df[Column.YEAR] == year]

    # Calcular la media de todas las columnas excepto 'Date', 'Year' y 'Month'
    agg_data = filtered_data.groupby([Column.YEAR, Column.MONTH])[
        filtered_data.columns.difference([Column.DATE, Column.YEAR, Column.MONTH])
    ].agg('mean').reset_index()

    # Crear una nueva columna Year-Month
    agg_data = agg_data.assign(Year_Month=agg_data[Column.YEAR].astype(str) + '-' + agg_data[Column.MONTH].astype(str))

    return agg_data



def ensure_directory_exists(base_path):
    """
    Verifica si el directorio especificado existe y lo crea si no existe.
    
    Args:
    base_path (str): Ruta del directorio a verificar/crear.
    """
    if not os.path.exists(base_path):
        try:
            os.makedirs(base_path)
            print(f"Directorio creado: {base_path}")
        except OSError as e:
            print(f"Error al crear el directorio {base_path}: {e}")
    else:
        print(f"El directorio ya existe: {base_path}")


def transform_variable(df: pd.DataFrame, x:str) -> pd.DataFrame:
    """
    Transforma una variable a numérica si no lo es ya.
    
    Args:
    df (pd.DataFrame): El DataFrame de entrada.
    x (str): El nombre de la columna a transformar.
    
    Returns:
    pd.Series: La serie transformada.
    """
    if isinstance(df[x][0], numbers.Number):
        return df[x]
    else:
        return pd.DataFrame([i for i in range(0, len(df[x]))])
    
def print_separator() -> None:
    """
    Imprime una línea separadora que consiste en 150 caracteres de 
    almohadilla ('#') tres veces.

    Esta función imprime tres líneas consecutivas, cada una de 150 
    caracteres de almohadilla ('#'), en la consola. Es útil para 
    separar visualmente secciones de salida en aplicaciones de consola.

    Retorna:
        None
    """
    print('#' * 150)
    print('#' * 150)
    print('#' * 150)