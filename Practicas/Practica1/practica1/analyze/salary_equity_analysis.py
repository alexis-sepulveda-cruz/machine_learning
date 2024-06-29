import pandas as pd
import matplotlib.pyplot as plt
from practica1.utils.util import ensure_directory_exists, print_tabulate
import seaborn as sns

class SalaryEquityAnalysis:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['Sueldo Neto'] = pd.to_numeric(self.df['Sueldo Neto'], errors='coerce')
        self.base_path = 'img/salary_equity'

        ensure_directory_exists(self.base_path)

    def calculate_deviation_by_dependency(self):
        desviacion_por_dependencia = self.df.groupby('dependencia')['Sueldo Neto'].agg(['mean', 'std']).reset_index()
        desviacion_por_dependencia['coef_variacion'] = desviacion_por_dependencia['std'] / desviacion_por_dependencia['mean']
        desviacion_por_dependencia = desviacion_por_dependencia.sort_values('coef_variacion', ascending=False)
        return desviacion_por_dependencia

    def plot_deviation_by_dependency(self, desviacion_por_dependencia):
        plt.figure(figsize=(12, 6))
        sns.barplot(x='dependencia', y='std', data=desviacion_por_dependencia.head(20))
        plt.title('Desviación Estándar de Sueldos por Dependencia (Top 20)')
        plt.xlabel('Dependencia')
        plt.ylabel('Desviación Estándar del Sueldo')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/desviacion_estandar_sueldos_por_dependencia.png')
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.barplot(x='dependencia', y='coef_variacion', data=desviacion_por_dependencia.head(20))
        plt.title('Coeficiente de Variación de Sueldos por Dependencia (Top 20)')
        plt.xlabel('Dependencia')
        plt.ylabel('Coeficiente de Variación')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/coeficiente_variacion_sueldos_por_dependencia.png')
        plt.close()

    def calculate_deviation_by_type(self):
        desviacion_por_tipo = self.df.groupby('Tipo')['Sueldo Neto'].agg(['mean', 'std']).reset_index()
        desviacion_por_tipo['coef_variacion'] = desviacion_por_tipo['std'] / desviacion_por_tipo['mean']
        desviacion_por_tipo = desviacion_por_tipo.sort_values('coef_variacion', ascending=False)
        return desviacion_por_tipo

    def plot_deviation_by_type(self, desviacion_por_tipo):
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Tipo', y='std', data=desviacion_por_tipo)
        plt.title('Desviación Estándar de Sueldos por Tipo de Edificio')
        plt.xlabel('Tipo de Edificio')
        plt.ylabel('Desviación Estándar del Sueldo')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/desviacion_estandar_sueldos_por_tipo.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Tipo', y='coef_variacion', data=desviacion_por_tipo)
        plt.title('Coeficiente de Variación de Sueldos por Tipo de Edificio')
        plt.xlabel('Tipo de Edificio')
        plt.ylabel('Coeficiente de Variación')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/coeficiente_variacion_sueldos_por_tipo.png')
        plt.close()

    def identify_disparity_areas(self, desviacion_por_dependencia):
        print("\nÁreas con mayor disparidad salarial (top 5):")
        print(desviacion_por_dependencia.head())

        print("\nÁreas con menor disparidad salarial (top 5):")
        print(desviacion_por_dependencia.tail())

    def plot_salary_disparity_relationship(self, desviacion_por_dependencia):
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='mean', y='coef_variacion', data=desviacion_por_dependencia)
        plt.title('Relación entre Sueldo Promedio y Disparidad Salarial por Dependencia')
        plt.xlabel('Sueldo Promedio')
        plt.ylabel('Coeficiente de Variación')
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/relacion_sueldo_promedio_disparidad.png')
        plt.close()

    def analyze(self):
        desviacion_por_dependencia = self.calculate_deviation_by_dependency()
        print("Desviación estándar y coeficiente de variación de sueldos por dependencia:")
        print_tabulate(desviacion_por_dependencia)
        self.plot_deviation_by_dependency(desviacion_por_dependencia)

        desviacion_por_tipo = self.calculate_deviation_by_type()
        print("\nDesviación estándar y coeficiente de variación de sueldos por tipo de edificio:")
        print_tabulate(desviacion_por_tipo)
        self.plot_deviation_by_type(desviacion_por_tipo)

        self.identify_disparity_areas(desviacion_por_dependencia)
        self.plot_salary_disparity_relationship(desviacion_por_dependencia)
