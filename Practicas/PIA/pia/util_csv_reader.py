import pandas as pd
from typing import Optional

class UtilCSVReader:
    def __init__(self, csv_folder: str = 'csv'):
        self.csv_folder = csv_folder

    def read_csv_to_dataframe(self, data_file: str) -> Optional[pd.DataFrame]:
        try:
            # Leer el archivo CSV
            df = pd.read_csv(f'{self.csv_folder}/{data_file}')
            
            # Aplicar conversiones
            df['id'] = df['id'].astype(int)
            df['title_word_count'] = df['title_word_count'].astype(int)
            df['document_entropy'] = df['document_entropy'].astype(float)
            df['freshness'] = df['freshness'].astype(int)
            df['easiness'] = df['easiness'].astype(float)
            df['fraction_stopword_presence'] = df['fraction_stopword_presence'].astype(float)
            df['normalization_rate'] = df['normalization_rate'].astype(float)
            df['speaker_speed'] = df['speaker_speed'].astype(float)
            df['silent_period_rate'] = df['silent_period_rate'].astype(float)
            
            # Verificar si existe la columna 'engagement' y si tiene valores faltantes
            if 'engagement' in df.columns and not df['engagement'].isnull().any():
                df['engagement'] = df['engagement'].astype(bool)
            
            return df
        except FileNotFoundError:
            print(f"El archivo {data_file} no se encontró en la carpeta {self.csv_folder}")
            return None
        except pd.errors.EmptyDataError:
            print(f"El archivo {data_file} está vacío")
            return None
        except Exception as e:
            print(f"Ocurrió un error al leer el archivo {data_file}: {str(e)}")
            return None