import pandas as pd
import matplotlib.pyplot as plt
from practica1.utils.util import ensure_directory_exists
import seaborn as sns
from scipy import stats

class CorrelationAnalysis:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['Sueldo Neto'] = pd.to_numeric(self.df['Sueldo Neto'], errors='coerce')
        self.base_path = 'img/correlation'

        ensure_directory_exists(self.base_path)

    def calculate_correlation_by_building_type(self):
        # Crear variables dummy para el tipo de edificio
        df_dummy = pd.get_dummies(self.df, columns=['Tipo'], prefix='Tipo')

        # Calcular la correlación entre el tipo de edificio y el sueldo neto
        correlaciones_tipo = df_dummy[['Sueldo Neto'] + [col for col in df_dummy.columns if col.startswith('Tipo_')]].corr()['Sueldo Neto'].sort_values(ascending=False)
        return correlaciones_tipo

    def plot_correlation_by_building_type(self, correlaciones_tipo):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=correlaciones_tipo.index, y=correlaciones_tipo.values)
        plt.title('Correlación entre Tipo de Edificio y Sueldo Neto')
        plt.xlabel('Tipo de Edificio')
        plt.ylabel('Coeficiente de Correlación')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/correlacion_tipo_edificio_sueldo.png')
        plt.close()

    def calculate_mean_salary_by_dependency(self):
        sueldo_promedio_dependencia = self.df.groupby('dependencia')['Sueldo Neto'].mean().sort_values(ascending=False)
        return sueldo_promedio_dependencia

    def plot_mean_salary_by_dependency(self, sueldo_promedio_dependencia):
        plt.figure(figsize=(12, 18))
        sns.stripplot(x=sueldo_promedio_dependencia.values, y=sueldo_promedio_dependencia.index, jitter=True)
        plt.title('Sueldo Neto Promedio por Dependencia')
        plt.xlabel('Sueldo Neto Promedio')
        plt.ylabel('Dependencia')
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/sueldo_promedio_por_dependencia_dotplot.png')
        plt.close()

    def perform_anova_by_dependency(self):
        dependencias = self.df['dependencia'].unique()
        sueldos_por_dependencia = [self.df[self.df['dependencia'] == dep]['Sueldo Neto'] for dep in dependencias]
        
        f_statistic, p_value = stats.f_oneway(*sueldos_por_dependencia)
        return f_statistic, p_value

    def analyze(self):
        correlaciones_tipo = self.calculate_correlation_by_building_type()
        print("Correlación entre el tipo de edificio y el sueldo neto:")
        print(correlaciones_tipo)
        self.plot_correlation_by_building_type(correlaciones_tipo)

        sueldo_promedio_dependencia = self.calculate_mean_salary_by_dependency()
        self.plot_mean_salary_by_dependency(sueldo_promedio_dependencia)

        f_statistic, p_value = self.perform_anova_by_dependency()
        print("\nResultados del análisis ANOVA para las dependencias:")
        print(f"Estadístico F: {f_statistic}")
        print(f"Valor p: {p_value}")

        if p_value < 0.05:
            print("Hay diferencias significativas en los sueldos entre las dependencias.")
        else:
            print("No hay diferencias significativas en los sueldos entre las dependencias.")
