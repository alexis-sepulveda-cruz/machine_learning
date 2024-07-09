import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from practica2.utils.util import ensure_directory_exists, transform_variable
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler


class AssetRegression:
    def __init__(self, data, base_path='img/regression'):
        self.data = data
        self.scaler = StandardScaler()
        self.base_path = base_path
        ensure_directory_exists(self.base_path)

    def filter_data_by_year(self, year) -> pd.DataFrame:
        """
        Filtra los datos por un año específico.
        """
        return self.data[self.data['Year'] == year]

    def train_model(self, X, y, model_class, model_params):
        """
        Entrena un modelo de regresión.
        """
        model = model_class(**model_params)
        model.fit(X, y)
        return model

    def plot_regression(self, df, x, y_columns, model_class, model_params_list, title_suffix):
        """
        Realiza una regresión lineal y grafica los resultados.
        """
        fig, ax = plt.subplots(figsize=(12, 8))


        for y in y_columns:
            X = transform_variable(df, x).values.reshape(-1, 1)
            y_data = df[y].values
            ax.scatter(df[x], y_data, marker='.', label=f'Datos de {y}')

            for params in model_params_list:
                model = self.train_model(X, y_data, model_class, params)
                y_pred = model.predict(X)
                ax.plot(df[x], y_pred, label=f"{model_class.__name__} {params} para {y}")

            mean = y_data.mean()
            ax.axhline(y=mean, linestyle='--', label=f'Media de {y}')

        ax.set_xlabel(x)
        ax.set_ylabel('Precio')
        ax.set_title(f'Regresión: {", ".join(y_columns)} vs {x} {title_suffix}')
        # Crear la leyenda
        # ax.legend(loc="upper left")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.tick_params(axis='x', rotation=90)
        save_path = f'{self.base_path}/{"_".join(y_columns)}_vs_{x}_{title_suffix}.png'

        # Truncar el nombre del archivo si es demasiado largo
        max_length = 255
        extension = '.png'
        base_name = os.path.basename(save_path)
        if len(base_name) > max_length:
            base_name = base_name[:max_length - len(extension)] + extension
            save_path = os.path.join(self.base_path, base_name)

        # Ajustar el espacio entre y alrededor del gráfico
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close(fig)
        print(f'Gráfico guardado en: {save_path}')

    def linear_regression(self, df, x, y):
        """
        Realiza una regresión lineal.
        """
        self.plot_regression(df, x, y, LinearRegression, [{}], 'Linear')

    def ridge_regression(self, df, x, y, alphas):
        """
        Realiza una regresión Ridge con diferentes valores de alpha.
        """
        model_params_list = [{'alpha': alpha} for alpha in alphas]
        self.plot_regression(df, x, y, Ridge, model_params_list, 'Ridge')

    def lasso_regression(self, df, x, y, alphas):
        """
        Realiza una regresión Lasso con diferentes valores de alpha.
        """
        model_params_list = [{'alpha': alpha} for alpha in alphas]
        self.plot_regression(df, x, y, Lasso, model_params_list, 'Lasso')
