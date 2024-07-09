import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from practica2.utils.util import ensure_directory_exists, transform_variable
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


class AssetRegression:
    def __init__(self, data, base_path='img/regression'):
        self.data = data
        self.scaler = StandardScaler()
        self.base_path = base_path
        ensure_directory_exists(self.base_path)

    def filter_data_by_year(self, year) -> pd.DataFrame:
        """Filtra los datos por un año específico."""
        return self.data[self.data['Year'] == year]

    def train_model(self, X, y, model_class, model_params):
        """Entrena un modelo de regresión."""
        model = model_class(**model_params)
        model.fit(X, y)
        return model

    def plot_regression(self, df, x, y_columns, model_class, model_params_list, title_suffix):
        """Realiza una regresión y grafica los resultados."""
        fig, ax = plt.subplots(figsize=(12, 8))
        X = transform_variable(df, x).values.reshape(-1, 1)
        
        for y in y_columns:
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
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.tick_params(axis='x', rotation=90)
        save_path = self._get_save_path(y_columns, x, title_suffix)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close(fig)
        print(f'Gráfico guardado en: {save_path}')

    def _get_save_path(self, y_columns, x, title_suffix):
        """Genera la ruta para guardar el gráfico, truncando si es necesario."""
        base_name = f'{"_".join(y_columns)}_vs_{x}_{title_suffix}.png'
        save_path = os.path.join(self.base_path, base_name)
        max_length = 255

        if len(os.path.basename(save_path)) > max_length:
            extension = '.png'
            truncated_name = base_name[:max_length - len(extension)] + extension
            save_path = os.path.join(self.base_path, truncated_name)

        return save_path

    def linear_regression(self, df, x, y):
        """Realiza una regresión lineal."""
        self.plot_regression(df, x, y, LinearRegression, [{}], 'Linear')

    def ridge_regression(self, df, x, y, alphas):
        """Realiza una regresión Ridge con diferentes valores de alpha."""
        model_params_list = [{'alpha': alpha} for alpha in alphas]
        self.plot_regression(df, x, y, Ridge, model_params_list, 'Ridge')

    def lasso_regression(self, df, x, y, alphas):
        """Realiza una regresión Lasso con diferentes valores de alpha."""
        model_params_list = [{'alpha': alpha} for alpha in alphas]
        self.plot_regression(df, x, y, Lasso, model_params_list, 'Lasso')

    def polynomial_regression(self, df, x, y, degrees):
        """Realiza una regresión polinomial con diferentes grados."""
        X = transform_variable(df, x).values.reshape(-1, 1)
        for degree in degrees:
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)
            self.plot_regression_with_poly(df, x, y, X_poly, LinearRegression, [{}], f'Polynomial (degree={degree})')

    def polynomial_ridge_regression(self, df, x, y, degrees, alphas):
        """Realiza una regresión polinomial con Ridge y diferentes valores de alpha."""
        X = transform_variable(df, x).values.reshape(-1, 1)
        for degree in degrees:
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)
            model_params_list = [{'alpha': alpha} for alpha in alphas]
            self.plot_regression_with_poly(df, x, y, X_poly, Ridge, model_params_list, f'Polynomial Ridge (degree={degree})')

    def polynomial_lasso_regression(self, df, x, y, degrees, alphas):
        """Realiza una regresión polinomial con Lasso y diferentes valores de alpha."""
        X = transform_variable(df, x).values.reshape(-1, 1)
        for degree in degrees:
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)
            model_params_list = [{'alpha': alpha} for alpha in alphas]
            self.plot_regression_with_poly(df, x, y, X_poly, Lasso, model_params_list, f'Polynomial Lasso (degree={degree})')

    def knn_regression(self, df, x, y, n_neighbors):
        """Realiza una regresión KNN con diferentes valores de n_neighbors."""
        model_params_list = [{'n_neighbors': n} for n in n_neighbors]
        self.plot_regression(df, x, y, KNeighborsRegressor, model_params_list, 'KNN')

    def decision_tree_regression(self, df, x, y, max_depths):
        """Realiza una regresión con árbol de decisión con diferentes profundidades máximas."""
        model_params_list = [{'max_depth': depth} for depth in max_depths]
        self.plot_regression(df, x, y, DecisionTreeRegressor, model_params_list, 'Decision Tree')

    def plot_regression_with_poly(self, df, x, y_columns, X_poly, model_class, model_params_list, title_suffix):
        """Grafica regresión polinomial."""
        fig, ax = plt.subplots(figsize=(12, 8))

        for y in y_columns:
            y_data = df[y].values
            ax.scatter(df[x], y_data, marker='.', label=f'Datos de {y}')

            for params in model_params_list:
                model = self.train_model(X_poly, y_data, model_class, params)
                y_pred = model.predict(X_poly)
                ax.plot(df[x], y_pred, label=f"{model_class.__name__} {params} para {y}")

            mean = y_data.mean()
            ax.axhline(y=mean, linestyle='--', label=f'Media de {y}')

        ax.set_xlabel(x)
        ax.set_ylabel('Precio')
        ax.set_title(f'Regresión: {", ".join(y_columns)} vs {x} {title_suffix}')
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.tick_params(axis='x', rotation=90)
        save_path = self._get_save_path(y_columns, x, title_suffix)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close(fig)
        print(f'Gráfico guardado en: {save_path}')