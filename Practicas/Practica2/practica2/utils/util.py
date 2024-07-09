import numbers
import os
from practica2.utils.column_assets import ColumnAssets as Column
import pandas as pd


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
    for col in df.columns:
        if col not in [Column.DATE, Column.YEAR, Column.MONTH]:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass

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