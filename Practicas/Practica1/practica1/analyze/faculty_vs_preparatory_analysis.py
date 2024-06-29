import pandas as pd
import matplotlib.pyplot as plt
from practica1.utils.util import ensure_directory_exists
import seaborn as sns
from scipy import stats

class FacultyVsPreparatoryAnalysis:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['Sueldo Neto'] = pd.to_numeric(self.df['Sueldo Neto'], errors='coerce')
        self.base_path = 'img/faculty_preparatory'

        ensure_directory_exists(self.base_path)

    def filter_data(self):
        self.facultades = self.df[self.df['Tipo'] == 'FACULTAD']
        self.preparatorias = self.df[self.df['Tipo'] == 'PREPARATORIA']

    def calculate_statistics(self):
        self.stats_facultades = self.facultades['Sueldo Neto'].describe()
        self.stats_preparatorias = self.preparatorias['Sueldo Neto'].describe()

        print("Estadísticas descriptivas de sueldos en facultades:")
        print(self.stats_facultades)
        print("\nEstadísticas descriptivas de sueldos en preparatorias:")
        print(self.stats_preparatorias)

    def plot_boxplot(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Tipo', y='Sueldo Neto', data=self.df[self.df['Tipo'].isin(['FACULTAD', 'PREPARATORIA'])])
        plt.title('Distribución de Sueldos: Facultades vs Preparatorias')
        plt.savefig(f'{self.base_path}/boxplot_facultades_vs_preparatorias.png')
        plt.close()

    def plot_histogram(self):
        plt.figure(figsize=(12, 6))
        sns.histplot(self.facultades['Sueldo Neto'], kde=True, label='Facultades', color='blue', alpha=0.5)
        sns.histplot(self.preparatorias['Sueldo Neto'], kde=True, label='Preparatorias', color='red', alpha=0.5)
        plt.title('Distribución de Sueldos: Facultades vs Preparatorias')
        plt.xlabel('Sueldo Neto')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.savefig(f'{self.base_path}/histograma_facultades_vs_preparatorias.png')
        plt.close()

    def test_normality(self):
        _, self.p_value_facultades = stats.normaltest(self.facultades['Sueldo Neto'])
        _, self.p_value_preparatorias = stats.normaltest(self.preparatorias['Sueldo Neto'])

        print(f"\nPrueba de normalidad para facultades: p-value = {self.p_value_facultades}")
        print(f"Prueba de normalidad para preparatorias: p-value = {self.p_value_preparatorias}")

    def perform_t_test(self):
        if self.p_value_facultades > 0.05 and self.p_value_preparatorias > 0.05:
            # Datos son normales, usar t-test
            self.t_stat, self.p_value = stats.ttest_ind(self.facultades['Sueldo Neto'], self.preparatorias['Sueldo Neto'])
            self.test_name = "t-student"
        else:
            # Datos no son normales, usar Mann-Whitney U
            self.t_stat, self.p_value = stats.mannwhitneyu(self.facultades['Sueldo Neto'], self.preparatorias['Sueldo Neto'])
            self.test_name = "Mann-Whitney U"

        print(f"\nResultados de la prueba {self.test_name}:")
        print(f"Estadístico: {self.t_stat}")
        print(f"Valor p: {self.p_value}")

        if self.p_value < 0.05:
            print("Hay una diferencia significativa en los sueldos entre facultades y preparatorias.")
        else:
            print("No hay una diferencia significativa en los sueldos entre facultades y preparatorias.")

    def compare_average_salaries(self):
        self.sueldo_promedio_facultades = self.facultades['Sueldo Neto'].mean()
        self.sueldo_promedio_preparatorias = self.preparatorias['Sueldo Neto'].mean()

        print(f"\nSueldo promedio en facultades: {self.sueldo_promedio_facultades:.2f}")
        print(f"Sueldo promedio en preparatorias: {self.sueldo_promedio_preparatorias:.2f}")

        self.diferencia_porcentual = ((self.sueldo_promedio_facultades - self.sueldo_promedio_preparatorias) / self.sueldo_promedio_preparatorias) * 100
        print(f"Diferencia porcentual: {self.diferencia_porcentual:.2f}%")

    def plot_average_salaries(self):
        plt.figure(figsize=(8, 6))
        sns.barplot(x=['Facultades', 'Preparatorias'], y=[self.sueldo_promedio_facultades, self.sueldo_promedio_preparatorias])
        plt.title('Comparación de Sueldos Promedio: Facultades vs Preparatorias')
        plt.ylabel('Sueldo Neto Promedio')
        plt.savefig(f'{self.base_path}/comparacion_sueldo_promedio_facultades_vs_preparatorias.png')
        plt.close()

    def calculate_variation(self):
        self.cv_facultades = self.facultades['Sueldo Neto'].std() / self.facultades['Sueldo Neto'].mean()
        self.cv_preparatorias = self.preparatorias['Sueldo Neto'].std() / self.preparatorias['Sueldo Neto'].mean()

        print(f"\nCoeficiente de variación en facultades: {self.cv_facultades:.4f}")
        print(f"Coeficiente de variación en preparatorias: {self.cv_preparatorias:.4f}")

    def plot_variation(self):
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Tipo', y='Sueldo Neto', data=self.df[self.df['Tipo'].isin(['FACULTAD', 'PREPARATORIA'])])
        plt.title('Dispersión de Sueldos: Facultades vs Preparatorias')
        plt.savefig(f'{self.base_path}/dispersion_sueldos_facultades_vs_preparatorias.png')
        plt.close()

    def analyze(self):
        self.filter_data()
        self.calculate_statistics()
        self.plot_boxplot()
        self.plot_histogram()
        self.test_normality()
        self.perform_t_test()
        self.compare_average_salaries()
        self.plot_average_salaries()
        self.calculate_variation()
        self.plot_variation()
