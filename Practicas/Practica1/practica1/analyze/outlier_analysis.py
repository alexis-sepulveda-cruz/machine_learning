import pandas as pd
import matplotlib.pyplot as plt
from practica1.utils.util import ensure_directory_exists, print_tabulate
import seaborn as sns
import numpy as np

class OutlierAnalysis:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['Sueldo Neto'] = pd.to_numeric(self.df['Sueldo Neto'], errors='coerce')
        self.outliers = None
        self.limite_inferior = None
        self.limite_superior = None
        self.base_path = 'img/outlier'

        ensure_directory_exists(self.base_path)

    def identificar_outliers(self, columna):
        Q1 = self.df[columna].quantile(0.25)
        Q3 = self.df[columna].quantile(0.75)
        IQR = Q3 - Q1
        self.limite_inferior = Q1 - 1.5 * IQR
        self.limite_superior = Q3 + 1.5 * IQR
        self.outliers = self.df[(self.df[columna] < self.limite_inferior) | (self.df[columna] > self.limite_superior)]
        return self.outliers

    def print_outlier_info(self):
        print(f"Número de outliers identificados: {len(self.outliers)}")
        print(f"Porcentaje de outliers: {(len(self.outliers) / len(self.df)) * 100:.2f}%")
        print(f"Límite inferior para outliers: {self.limite_inferior}")
        print(f"Límite superior para outliers: {self.limite_superior}")

    def plot_distribution_with_outliers(self):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Sueldo Neto', data=self.df)
        plt.title('Distribución de Sueldos con Outliers')
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/distribucion_sueldos_con_outliers.png')
        plt.close()

    def plot_outliers_scatter(self):
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=self.df.index, y='Sueldo Neto', data=self.df)
        plt.axhline(y=self.limite_superior, color='r', linestyle='--', label='Límite Superior')
        plt.axhline(y=self.limite_inferior, color='r', linestyle='--', label='Límite Inferior')
        plt.title('Outliers en Sueldos')
        plt.xlabel('Índice')
        plt.ylabel('Sueldo Neto')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/outliers_sueldos.png')
        plt.close()

    def analyze_outliers_by_dependency(self):
        outliers_por_dependencia = self.outliers.groupby('dependencia').size().sort_values(ascending=False)
        print("\nDependencias con mayor número de outliers:")
        print_tabulate(outliers_por_dependencia.head())

        plt.figure(figsize=(12, 6))
        sns.barplot(x=outliers_por_dependencia.index[:10], y=outliers_por_dependencia.values[:10])
        plt.title('Top 10 Dependencias con Mayor Número de Outliers')
        plt.xlabel('Dependencia')
        plt.ylabel('Número de Outliers')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/outliers_por_dependencia.png')
        plt.close()

    def analyze_outliers_by_building_type(self):
        outliers_por_tipo = self.outliers.groupby('Tipo').size().sort_values(ascending=False)
        print("\nTipos de edificio con mayor número de outliers:")
        print_tabulate(outliers_por_tipo)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=outliers_por_tipo.index, y=outliers_por_tipo.values)
        plt.title('Número de Outliers por Tipo de Edificio')
        plt.xlabel('Tipo de Edificio')
        plt.ylabel('Número de Outliers')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/outliers_por_tipo_edificio.png')
        plt.close()

    def print_outlier_statistics(self):
        estadisticas_outliers = self.outliers['Sueldo Neto'].describe()
        print("\nEstadísticas descriptivas de los outliers:")
        print_tabulate(estadisticas_outliers)

    def compare_distribution_with_and_without_outliers(self):
        df_sin_outliers = self.df[(self.df['Sueldo Neto'] >= self.limite_inferior) & (self.df['Sueldo Neto'] <= self.limite_superior)]

        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.df, x='Sueldo Neto', kde=True, color='blue', alpha=0.5, label='Con Outliers')
        sns.histplot(data=df_sin_outliers, x='Sueldo Neto', kde=True, color='red', alpha=0.5, label='Sin Outliers')
        plt.title('Distribución de Sueldos Con y Sin Outliers')
        plt.xlabel('Sueldo Neto')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/distribucion_sueldos_con_sin_outliers.png')
        plt.close()

    def analyze(self):
        self.identificar_outliers('Sueldo Neto')
        self.print_outlier_info()
        self.plot_distribution_with_outliers()
        self.plot_outliers_scatter()
        self.analyze_outliers_by_dependency()
        self.analyze_outliers_by_building_type()
        self.print_outlier_statistics()
        self.compare_distribution_with_and_without_outliers()
