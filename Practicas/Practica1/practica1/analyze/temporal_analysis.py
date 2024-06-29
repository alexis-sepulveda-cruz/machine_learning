import pandas as pd
import matplotlib.pyplot as plt
from practica1.utils.util import ensure_directory_exists
import seaborn as sns

class TemporalAnalysis:

    def __init__(self, file_path):
        # Cargar el dataset
        self.df = pd.read_csv(file_path)

        # Convertir la columna de Sueldo Neto a tipo numérico
        self.df['Sueldo Neto'] = pd.to_numeric(self.df['Sueldo Neto'], errors='coerce')

        # Convertir la columna Fecha a tipo datetime
        self.df['Fecha'] = pd.to_datetime(self.df['Fecha'])

        # Crear columnas de año y mes
        self.df['year'] = self.df['Fecha'].dt.year
        self.df['month'] = self.df['Fecha'].dt.month
        self.base_path = 'img/temporal'

        ensure_directory_exists(self.base_path)

    def calculate_monthly_average(self) -> pd.DataFrame:
        # Calcular el sueldo promedio por mes y año
        sueldo_promedio = self.df.groupby(['year', 'month'])['Sueldo Neto'].mean().reset_index()
        sueldo_promedio['Fecha'] = pd.to_datetime(sueldo_promedio[['year', 'month']].assign(day=1))
        return sueldo_promedio

    def plot_monthly_evolution(self, sueldo_promedio: pd.DataFrame) -> None:
        # Crear una gráfica de línea para mostrar la evolución de los sueldos a lo largo del tiempo
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Fecha', y='Sueldo Neto', data=sueldo_promedio)
        plt.title('Evolución del Sueldo Promedio a lo largo del tiempo')
        plt.xlabel('Fecha')
        plt.ylabel('Sueldo Neto Promedio')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/evolucion_sueldo_promedio.png')
        plt.close()

    def identify_seasonal_trends(self) -> None:
        # Identificar tendencias estacionales
        sueldo_promedio_mensual = self.df.groupby('month')['Sueldo Neto'].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x='month', y='Sueldo Neto', data=sueldo_promedio_mensual)
        plt.title('Sueldo Promedio por Mes')
        plt.xlabel('Mes')
        plt.ylabel('Sueldo Neto Promedio')
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/sueldo_promedio_mensual.png')
        plt.close()

    def yearly_statistics(self) -> pd.DataFrame:
        # Calcular estadísticas descriptivas por año
        stats_por_año = self.df.groupby('year')['Sueldo Neto'].agg(['mean', 'min', 'max', 'std']).reset_index()
        print("Estadísticas descriptivas por año:")
        print(stats_por_año)
        return stats_por_año

    def plot_yearly_distribution(self):
        # Crear un gráfico de caja para visualizar la distribución de sueldos por año
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='year', y='Sueldo Neto', data=self.df)
        plt.title('Distribución de Sueldos por Año')
        plt.xlabel('Año')
        plt.ylabel('Sueldo Neto')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/boxplot_sueldos_por_año.png')
        plt.close()

    def calculate_annual_growth_rate(self) -> pd.DataFrame:
        # Calcular la tasa de crecimiento anual del sueldo promedio
        sueldo_promedio_anual = self.df.groupby('year')['Sueldo Neto'].mean().reset_index()
        sueldo_promedio_anual['Tasa_Crecimiento'] = sueldo_promedio_anual['Sueldo Neto'].pct_change() * 100
        print("\nTasa de crecimiento anual del sueldo promedio:")
        print(sueldo_promedio_anual)
        return sueldo_promedio_anual

    def plot_annual_growth_rate(self, sueldo_promedio_anual) -> None:
        # Visualizar la tasa de crecimiento anual
        plt.figure(figsize=(10, 6))
        sns.barplot(x='year', y='Tasa_Crecimiento', data=sueldo_promedio_anual)
        plt.title('Tasa de Crecimiento Anual del Sueldo Promedio')
        plt.xlabel('Año')
        plt.ylabel('Tasa de Crecimiento (%)')
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/tasa_crecimiento_anual.png')
        plt.close()

    def analyze(self) -> None:
        sueldo_promedio = self.calculate_monthly_average()
        self.plot_monthly_evolution(sueldo_promedio)
        self.identify_seasonal_trends()
        self.yearly_statistics()
        self.plot_yearly_distribution()
        sueldo_promedio_anual = self.calculate_annual_growth_rate()
        self.plot_annual_growth_rate(sueldo_promedio_anual)
