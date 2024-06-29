import pandas as pd
import matplotlib.pyplot as plt
from practica1.utils.util import ensure_directory_exists
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import List

class SalaryAnalysisByBuilding:
    def __init__(self, file_path):
        # Cargar el dataset
        self.df = pd.read_csv(file_path)

        # Convertir la columna de Sueldo Neto a tipo numérico
        self.df['Sueldo Neto'] = pd.to_numeric(self.df['Sueldo Neto'], errors='coerce')
        self.base_path = 'img/salary_building'

        ensure_directory_exists(self.base_path)

    def descriptive_statistics(self) -> pd.DataFrame:
        # Calcular estadísticas descriptivas por tipo de edificio
        stats_por_tipo = self.df.groupby('Tipo')['Sueldo Neto'].agg(['mean', 'min', 'max', 'std']).reset_index()
        print("Estadísticas descriptivas por tipo de edificio:")
        print(stats_por_tipo)
        return stats_por_tipo

    def plot_histograms(self) -> None:
        # Crear un histograma de distribución de sueldos para cada tipo de edificio
        plt.figure(figsize=(12, 8))
        for tipo in self.df['Tipo'].unique():
            sns.histplot(self.df[self.df['Tipo'] == tipo]['Sueldo Neto'], kde=True, label=tipo)
        plt.title('Distribución de Sueldos por Tipo de Edificio')
        plt.xlabel('Sueldo Neto')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.savefig(f'{self.base_path}/histograma_sueldos_por_tipo.png')
        plt.close()

    def normality_tests(self) -> any:
        # Realizar prueba de normalidad para cada grupo
        normal_tests = {}
        for tipo in self.df['Tipo'].unique():
            _, p_value = stats.normaltest(self.df[self.df['Tipo'] == tipo]['Sueldo Neto'])
            normal_tests[tipo] = p_value > 0.05
        return normal_tests

    def perform_anova(self, sueldos_por_tipo: pd.Series):
        f_statistic, p_value = stats.f_oneway(*sueldos_por_tipo)
        print("\nResultados de ANOVA:")
        print(f"Estadístico F: {f_statistic}")
        print(f"Valor p: {p_value}")
        
        if p_value < 0.05:
            print("Hay diferencias significativas entre los tipos de edificio.")
            self.perform_t_tests(sueldos_por_tipo)
        else:
            print("No hay diferencias significativas entre los tipos de edificio.")

    def perform_t_tests(self, sueldos_por_tipo: pd.Series) -> None:
        # Realizar pruebas t-student para comparaciones por pares
        tipos = self.df['Tipo'].unique()
        for i in range(len(tipos)):
            for j in range(i+1, len(tipos)):
                t_stat, t_p_value = stats.ttest_ind(sueldos_por_tipo[i], sueldos_por_tipo[j])
                print(f"\nComparación entre {tipos[i]} y {tipos[j]}:")
                print(f"Estadístico t: {t_stat}")
                print(f"Valor p: {t_p_value}")

    def perform_kruskal_wallis(self, sueldos_por_tipo: pd.Series) -> None:
        # Realizar prueba de Kruskal-Wallis
        h_statistic, p_value = stats.kruskal(*sueldos_por_tipo)
        print("\nResultados de la prueba de Kruskal-Wallis:")
        print(f"Estadístico H: {h_statistic}")
        print(f"Valor p: {p_value}")
        
        if p_value < 0.05:
            # Realizar prueba de Tukey
            print("Hay diferencias significativas entre los tipos de edificio.")
            tukey_results = pairwise_tukeyhsd(self.df['Sueldo Neto'], self.df['Tipo'])
            print("\nResultados de la prueba de Tukey:")
            print(tukey_results)
        else:
            print("No hay diferencias significativas entre los tipos de edificio.")

    def plot_boxplot(self) -> None:
        """
        Crear un gráfico de caja para visualizar la distribución 
        de sueldos por tipo de edificio
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Tipo', y='Sueldo Neto', data=self.df)
        plt.title('Distribución de Sueldos por Tipo de Edificio')
        plt.xlabel('Tipo de Edificio')
        plt.ylabel('Sueldo Neto')
        plt.xticks(rotation=45)
        plt.savefig(f'{self.base_path}/boxplot_sueldos_por_tipo.png')
        plt.close()

    def analyze(self) -> None:
        self.descriptive_statistics()
        self.plot_histograms()

        normal_tests = self.normality_tests()
        all_normal = all(normal_tests.values()) # Verificar si todos los grupos son normales
        tipos = self.df['Tipo'].unique()
        sueldos_por_tipo = [self.df[self.df['Tipo'] == tipo]['Sueldo Neto'] for tipo in tipos]

        if all_normal:
            # Realizar ANOVA
            self.perform_anova(sueldos_por_tipo)
        else:
            self.perform_kruskal_wallis(sueldos_por_tipo)

        self.plot_boxplot()
