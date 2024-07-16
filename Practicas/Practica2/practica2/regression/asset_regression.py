import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from practica2.utils.util import ensure_directory_exists, transform_variable
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


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

    def train_and_evaluate_model(self, df, x, y, model_class=LinearRegression, model_params={}):
        """
        Realiza una regresión lineal, evalúa el modelo en conjuntos de entrenamiento y prueba,
        y visualiza los resultados.
        """
        X = transform_variable(df, x).values.reshape(-1, 1)
        y_data = df[y]

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.2, random_state=42)

        # Entrenar el modelo
        model = self.train_model(X_train, y_train, model_class, model_params)

        # Realizar predicciones
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        print(f"Modelo: {model_class.__name__} con parámetros {model_params}")
        print(f"Entrenamiento - MSE: {mean_squared_error(y_train, y_train_pred)}, R2: {r2_score(y_train, y_train_pred)}")
        print(f"Prueba - MSE: {mean_squared_error(y_test, y_test_pred)}, R2: {r2_score(y_test, y_test_pred)}")

        # Evaluar el modelo
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Imprimir las métricas
        print(f'Train RMSE: {train_rmse}')
        print(f'Test RMSE: {test_rmse}')
        print(f'Train R2: {train_r2}')
        print(f'Test R2: {test_r2}')

        # Graficar resultados
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(X_train, y_train, color='blue', label='Train data')
        ax.scatter(X_test, y_test, color='green', label='Test data')
        ax.plot(X_train, y_train_pred, color='black', label='Train fit')
        ax.plot(X_test, y_test_pred, color='red', label='Test fit')

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f'{model_class.__name__} Regression: {y} vs {x}')
        ax.legend(loc='upper left')

        save_path = f'{self.base_path}/{y}_vs_{x}_{model_class.__name__}_train_test.png'
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close(fig)
        print(f'Gráfico guardado en: {save_path}')

    def train_and_evaluate_ridge(self, df, x, y, alpha=1.0):
        self.train_and_evaluate_model(df, x, y, model_class=Ridge, model_params={'alpha': alpha})

    def train_and_evaluate_lasso(self, df, x, y, alpha=1.0):
        self.train_and_evaluate_model(df, x, y, model_class=Lasso, model_params={'alpha': alpha})

    def train_and_evaluate_polynomial(self, df, x, y, degree=2):
        X = transform_variable(df, x).values.reshape(-1, 1)
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        poly_df = pd.DataFrame(X_poly, columns=[f'{x}^{i}' for i in range(degree+1)])
        poly_df[y] = df[y].values


        for colum in poly_df.columns.difference([y]):
            self.train_and_evaluate_model(poly_df, colum, y, model_class=LinearRegression)

    def train_and_evaluate_polynomial_ridge(self, df, x, y, degree=2, alpha=1.0):
        X = transform_variable(df, x).values.reshape(-1, 1)
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        poly_df = pd.DataFrame(X_poly, columns=[f'{x}^{i}' for i in range(degree+1)])
        poly_df[y] = df[y].values

        for colum in poly_df.columns.difference([y]):
            self.train_and_evaluate_model(poly_df, colum, y, model_class=Ridge, model_params={'alpha': alpha})

    def train_and_evaluate_polynomial_lasso(self, df, x, y, degree=2, alpha=1.0):
        X = transform_variable(df, x).values.reshape(-1, 1)
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        poly_df = pd.DataFrame(X_poly, columns=[f'{x}^{i}' for i in range(degree+1)])
        poly_df[y] = df[y].values

        for colum in poly_df.columns.difference([y]):
            self.train_and_evaluate_model(poly_df, colum, y, model_class=Lasso, model_params={'alpha': alpha})
    
    def train_and_evaluate_knn(self, df, x, y, n_neighbors=5):
        self.train_and_evaluate_model(df, x, y, model_class=KNeighborsRegressor, model_params={'n_neighbors': n_neighbors})

    def train_and_evaluate_decision_tree(self, df, x, y, max_depth=None):
        self.train_and_evaluate_model(df, x, y, model_class=DecisionTreeRegressor, model_params={'max_depth': max_depth})