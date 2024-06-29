import pandas as pd
import matplotlib.pyplot as plt
from practica1.enums.tipos_edificios_enum import TipoEdificio
import seaborn as sns
from practica1.utils.util import ensure_directory_exists, print_tabulate

class DependencyAnalysis:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['Sueldo Neto'] = pd.to_numeric(self.df['Sueldo Neto'], errors='coerce')
        self.base_path = 'img/dependency'
        ensure_directory_exists(self.base_path)

    def analyze_by_building_type(self):
        for tipo in TipoEdificio.obtener_atributos():
            df_tipo = self.df[self.df['Tipo'] == tipo]
            if df_tipo.empty:
                continue
            
            print(f"\n--- Análisis para {tipo} ---")
            self.analyze_dependencies(df_tipo, tipo)

    def analyze_dependencies(self, df, tipo):
        num_dependencias = df['dependencia'].nunique()
        stats = self.calculate_statistics(df, tipo, num_dependencias)
        self.plot_boxplot(df, tipo, num_dependencias)
        self.identify_top_bottom_dependencies(stats, tipo, num_dependencias)
        self.plot_top_dependencies(stats, tipo, num_dependencias)
        proporcion_empleados = self.calculate_employee_proportion(df, tipo)
        self.plot_employee_proportion(df, tipo, num_dependencias)
        self.plot_employee_salary_relation(stats, proporcion_empleados, tipo)

    def calculate_statistics(self, df, tipo, num_dependencias) -> pd.DataFrame:
        stats = df.groupby('dependencia')['Sueldo Neto'].agg(['count', 'mean', 'median', 'min', 'max', 'std']).reset_index()
        stats = stats.sort_values('mean', ascending=False)
        print(f"Estadísticas descriptivas para {tipo}:")
        print_tabulate(stats.head(min(5, num_dependencias)))
        print_tabulate(stats.tail(min(5, num_dependencias)))
        return stats

    def plot_boxplot(self, df, tipo, num_dependencias) -> None:
        if num_dependencias > 1:
            plt.figure(figsize=(20, 10))
            sns.boxplot(x='dependencia', y='Sueldo Neto', data=df)
            plt.title(f'Distribución de Sueldos por Dependencia - {tipo}')
            plt.xlabel('Dependencia')
            plt.ylabel('Sueldo Neto')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f'{self.base_path}/boxplot_sueldos_{tipo}.png')
            plt.close()
        else:
            print(f"No se genera boxplot para {tipo} debido a que solo hay una dependencia.")

    def identify_top_bottom_dependencies(self, stats, tipo, num_dependencias) -> None:
        if num_dependencias > 1:
            top_n = min(5, num_dependencias)
            top = stats.head(top_n)
            bottom = stats.tail(top_n)
            print(f"\nTop {top_n} dependencias con sueldos más altos en {tipo}:")
            print_tabulate(top[['dependencia', 'mean']])
            print(f"\nTop {top_n} dependencias con sueldos más bajos en {tipo}:")
            print_tabulate(bottom[['dependencia', 'mean']])
        else:
            print(f"\nEstadísticas para la única dependencia en {tipo}:")
            print_tabulate(stats[['dependencia', 'mean']])

    def plot_top_dependencies(self, stats, tipo, num_dependencias):
        if num_dependencias > 1:
            top_n = min(10, num_dependencias)
            plt.figure(figsize=(12, 6))
            sns.barplot(x='dependencia', y='mean', data=stats.head(top_n))
            plt.title(f'Top {top_n} Dependencias con Sueldos Promedio más Altos - {tipo}')
            plt.xlabel('Dependencia')
            plt.ylabel('Sueldo Neto Promedio')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{self.base_path}/top_dependencias_sueldos_altos_{tipo}.png')
            plt.close()
        else:
            print(f"No se genera gráfico de barras para {tipo} debido a que solo hay una dependencia.")

    def calculate_employee_proportion(self, df, tipo):
        proporcion = df['dependencia'].value_counts(normalize=True) * 100
        proporcion = proporcion.reset_index()
        proporcion.columns = ['dependencia', 'proporcion_empleados']
        return proporcion

    def plot_employee_proportion(self, df, tipo, num_dependencias):
        if num_dependencias > 1:
            proporcion = df['dependencia'].value_counts(normalize=True) * 100
            plt.figure(figsize=(12, 8))
            proporcion.head(min(10, num_dependencias)).plot(kind='pie', autopct='%1.1f%%')
            plt.title(f'Proporción de Empleados por Dependencia - {tipo}')
            plt.ylabel('')
            plt.tight_layout()
            plt.savefig(f'{self.base_path}/proporcion_empleados_{tipo}.png')
            plt.close()
        else:
            print(f"No se genera gráfico de proporción para {tipo} debido a que solo hay una dependencia.")

    def plot_employee_salary_relation(self, stats, proporcion_empleados, tipo):
        relacion = pd.merge(stats, proporcion_empleados, on='dependencia')
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='count', y='mean', size='proporcion_empleados', 
                        hue='proporcion_empleados', palette='viridis', 
                        data=relacion, legend='brief', sizes=(20, 200))
        plt.title(f'Relación entre Número de Empleados y Sueldo Promedio - {tipo}')
        plt.xlabel('Número de Empleados')
        plt.ylabel('Sueldo Promedio')
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/relacion_empleados_sueldo_{tipo}.png')
        plt.close()

    def analyze(self):
        self.analyze_by_building_type()