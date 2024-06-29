import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from practica1.utils.util import ensure_directory_exists, material_colors

class EmployeeDistributionAnalysis:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.base_path = 'img/employee'

        ensure_directory_exists(self.base_path)

    def calculate_employees_by_type(self):
        empleados_por_tipo = self.df['Tipo'].value_counts()
        return empleados_por_tipo

    def plot_pie_chart(self, empleados_por_tipo) -> None:
        plt.figure(figsize=(10, 8))
        plt.pie(empleados_por_tipo.values, labels=empleados_por_tipo.index, autopct='%1.1f%%', startangle=90)
        plt.title('Distribución de Empleados por Tipo de Edificio')
        plt.axis('equal')  # Para asegurar que el pie sea circular
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/distribucion_empleados_por_tipo.png')
        plt.close()

    def calculate_admin_proportion(self):
        self.df['Es_Admin'] = self.df['Tipo'] == 'ADMIN'
        proporcion_admin = self.df['Es_Admin'].value_counts(normalize=True) * 100
        print("Proporción de empleados en roles administrativos vs. otros roles:")
        print(f"Administrativos: {proporcion_admin[True]:.2f}%")
        print(f"Otros roles: {proporcion_admin[False]:.2f}%")
        return proporcion_admin

    def plot_admin_proportion(self, proporcion_admin) -> None:
        plt.figure(figsize=(8, 6))
        sns.barplot(x=['Administrativos', 'Otros roles'], y=proporcion_admin.values)
        plt.title('Proporción de Empleados Administrativos vs. Otros Roles')
        plt.ylabel('Porcentaje')
        plt.ylim(0, 100)
        for i, v in enumerate(proporcion_admin.values):
            plt.text(i, v + 1, f'{v:.2f}%', ha='center')
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/proporcion_admin_vs_otros.png')
        plt.close()

    def calculate_employees_by_dependency(self):
        empleados_por_dependencia = self.df['dependencia'].value_counts()
        return empleados_por_dependencia

    def plot_top_20_dependencies(self, empleados_por_dependencia) -> None:
        plt.figure(figsize=(12, 10))
        empleados_por_dependencia.head(20).plot(kind='barh')
        plt.title('Top 20 Dependencias con Mayor Número de Empleados')
        plt.xlabel('Número de Empleados')
        plt.ylabel('Dependencia')
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/top_20_dependencias_por_empleados.png')
        plt.close()

    def calculate_distribution_by_type_and_dependency(self) -> pd.DataFrame:
        distribucion_tipo_dependencia = self.df.groupby(['Tipo', 'dependencia']).size().unstack(fill_value=0)
        return distribucion_tipo_dependencia

    def plot_heatmap(self, distribucion_tipo_dependencia) -> None:
        plt.figure(figsize=(16, 12))
        sns.heatmap(distribucion_tipo_dependencia, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Distribución de Empleados por Tipo de Edificio y Dependencia')
        plt.xlabel('Dependencia')
        plt.ylabel('Tipo de Edificio')
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/distribucion_empleados_tipo_dependencia.png')
        plt.close()

    def plot_stacked_bar_chart(self, distribucion_tipo_dependencia) -> None:
        proporcion_tipo_dependencia = distribucion_tipo_dependencia.div(distribucion_tipo_dependencia.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(15, 33))
        proporcion_tipo_dependencia.plot(kind='bar', stacked=True, ax=ax, color=material_colors)
        ax.set_title('Proporción de Tipos de Edificio por Dependencia', fontsize=20)
        ax.set_xlabel('Tipo de Edificio', fontsize=16)
        ax.set_ylabel('Proporción', fontsize=16)
        ax.legend(title='Dependencia', bbox_to_anchor=(1, 1), loc='upper left')
        plt.subplots_adjust(left=0.1, right=1.8, top=0.9, bottom=0.2)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/proporcion_tipo_edificio_por_dependencia.png')
        plt.close()

    def analyze(self) -> None:
        empleados_por_tipo = self.calculate_employees_by_type()
        self.plot_pie_chart(empleados_por_tipo)
        proporcion_admin = self.calculate_admin_proportion()
        self.plot_admin_proportion(proporcion_admin)
        empleados_por_dependencia = self.calculate_employees_by_dependency()
        self.plot_top_20_dependencies(empleados_por_dependencia)
        distribucion_tipo_dependencia = self.calculate_distribution_by_type_and_dependency()
        self.plot_heatmap(distribucion_tipo_dependencia)
        self.plot_stacked_bar_chart(distribucion_tipo_dependencia)
