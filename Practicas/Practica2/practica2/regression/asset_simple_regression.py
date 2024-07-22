import os
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from practica2.utils.util import ensure_directory_exists, transform_variable
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


class AssetSimpleRegression:

    def __init__(self, data: pd.DataFrame, base_path: str = 'img/simple-regression'):
        """
        Inicializa la clase AssetRegression.

        Args:
            data (pd.DataFrame): Los datos de activos.
            base_path (str, optional): Ruta base para guardar 
                                       las imágenes. Por defecto 
                                       es 'img/regression'.
        """
        self.data = data
        self.scaler = StandardScaler()
        self.base_path = base_path
        ensure_directory_exists(self.base_path)
    
    def filter_data_by_year(self, year: int) -> pd.DataFrame:
        """
        Filtra los datos por año.

        Args:
            year (int): El año para filtrar los datos.

        Returns:
            pd.DataFrame: Los datos filtrados para el año especificado.
        """
        return self.data[self.data['Year'] == year]
    
    def _get_save_path(self, y_columns: list, x: str, title_suffix: str) -> str:
        """
        Genera la ruta para guardar el gráfico, truncando si es necesario.

        Args:
            y_columns (list): Lista de nombres de columnas para las variables 
                              dependientes.
            x (str): Nombre de la columna para la variable 
                     independiente.
            title_suffix (str): Sufijo para el título del gráfico.

        Returns:
            str: La ruta para guardar el gráfico.
        """
        base_name = f'{"_".join(y_columns)}_vs_{x}_{title_suffix}.png'
        save_path = os.path.join(self.base_path, base_name)
        max_length = 255

        if len(os.path.basename(save_path)) > max_length:
            extension = '.png'
            truncated_name = base_name[:max_length - len(extension)] + extension
            save_path = os.path.join(self.base_path, truncated_name)

        return save_path

    def plot_regression(
        self,
        df: pd.DataFrame,
        x: str,
        y_columns: list,
        model_class: Tuple[
            LinearRegression, Lasso, Lasso, 
            KNeighborsRegressor, DecisionTreeRegressor
        ],
        model_params_list: list,
        title_suffix: str
    ) -> None:
        """
        Grafica la regresión para una o más variables dependientes.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y_columns (list): Lista de nombres de columnas para las variables dependientes.
            model_class (type): Clase del modelo a utilizar.
            model_params_list (list): Lista de diccionarios con parámetros para el modelo.
            title_suffix (str): Sufijo para el título del gráfico.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        X = transform_variable(df, x).values.reshape(-1, 1)
        
        for y in y_columns:
            y_data = df[y].values
            ax.scatter(df[x], y_data, marker='.', label=f'Datos de {y}')

            for params in model_params_list:
                model = self.train_model(X, y_data, model_class, params)
                y_pred = model.predict(X)
                ax.plot(df[x], y_pred, label=f"{model_class.__name__} {params} para {y}")
                #print(f"alpha: {params}, coeficientes: {model.coef_[0]:.4f}, intercepto: {model.intercept_:.4f}")
                print(f"Alpha: {params}")
                if hasattr(model, 'coef_'):
                    print(f"Coeficientes: {model.coef_[0]:.4f}")
                if hasattr(model, 'intercept_'):
                    print(f"Intercepto: {model.intercept_:.4f}")
                print(f"Media de {y}: {y_data.mean():.4f}")
                print("--------------------")

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

    def linear_regression(self, df: pd.DataFrame, x: str, y: list) -> None:
        """
        Realiza y grafica una regresión lineal.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (list): Lista de nombres de columnas para las variables 
                      dependientes.
        """
        self.plot_regression(
            df=df, x=x, y=y, model_class=LinearRegression, 
            model_params_list=[{}], title_suffix='Linear'
        )

    def ridge_regression(self, df: pd.DataFrame, x: str, y: list, alphas: list) -> None:
        """
        Realiza y grafica una regresión Ridge.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (list): Lista de nombres de columnas para las variables 
                      dependientes.
            alphas (list): Lista de valores alpha para la regresión Ridge.
        """
        model_params_list = [{'alpha': alpha} for alpha in alphas]
        self.plot_regression(
            df=df, x=x, y=y, model_class=Ridge, 
            model_params_list=model_params_list, title_suffix='Ridge'
        )

    def lasso_regression(self, df: pd.DataFrame, x: str, y: list, alphas: list) -> None:
        """
        Realiza y grafica una regresión Lasso.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (list): Lista de nombres de columnas para las variables 
                      dependientes.
            alphas (list): Lista de valores alpha para la regresión Lasso.
        """
        model_params_list = [{'alpha': alpha} for alpha in alphas]
        self.plot_regression(
            df=df, x=x, y=y, model_class=Lasso, 
            model_params_list=model_params_list, title_suffix='Lasso'
        )

    def knn_regression(self, df: pd.DataFrame, x: str, y: list, n_neighbors: list) -> None:
        """
        Realiza y grafica una regresión KNN.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (list): Lista de nombres de columnas para las variables dependientes.
            n_neighbors (list): Lista de valores para el número de vecinos.
        """
        model_params_list = [{'n_neighbors': n} for n in n_neighbors]
        self.plot_regression(
            df=df, x=x, y=y, model_class=KNeighborsRegressor, 
            model_params_list=model_params_list, title_suffix='KNN'
        )

    def decision_tree_regression(
        self,
        df: pd.DataFrame,
        x: str,
        y: list,
        max_depths: list
    ) -> None:
        """
        Realiza y grafica una regresión con árbol de decisión.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (list): Lista de nombres de columnas para las variables dependientes.
            max_depths (list): Lista de valores para la profundidad máxima del árbol.
        """
        model_params_list = [{'max_depth': depth} for depth in max_depths]
        self.plot_regression(
            df=df, x=x, y=y, model_class=DecisionTreeRegressor, 
            model_params_list=model_params_list, title_suffix='Decision Tree'
        )

    def plot_regression_with_poly(
        self,
        df: pd.DataFrame,
        x: str,
        y_columns: list,
        X_poly: np.ndarray,
        model_class: Tuple[LinearRegression, Ridge, Lasso],
        model_params_list: list,
        title_suffix: str,
        degree: int
    ) -> None:
        """
        Grafica la regresión polinomial para una o más variables dependientes.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y_columns (list): Lista de nombres de columnas para las variables 
                              dependientes.
            X_poly (np.ndarray): Características polinomiales.
            model_class (type): Clase del modelo a utilizar.
            model_params_list (list): Lista de diccionarios con parámetros para 
                                      el modelo.
            title_suffix (str): Sufijo para el título del gráfico.
            degree (int): Grado para la regresión polinomial.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        for y in y_columns:
            y_data = df[y].values
            ax.scatter(df[x], y_data, marker='.', label=f'Datos de {y}')

            for params in model_params_list:
                model = self.train_model(X_poly, y_data, model_class, params)
                y_pred = model.predict(X_poly)
                ax.plot(df[x], y_pred, label=f"{model_class.__name__} {params} para {y}")
                print(f"Grado: {degree}")
                print(f"Coeficientes: {model.coef_}")
                print(f"Intercepto: {model.intercept_:.4f}")
                print("--------------------")

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

    def polynomial_regression(
        self,
        df: pd.DataFrame,
        x: str,
        y: list,
        degrees: list
    ) -> None:
        """
        Realiza y grafica una regresión polinomial.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (list): Lista de nombres de columnas para las variables 
            dependientes.
            degrees (list): Lista de grados para la regresión polinomial.
        """
        X = transform_variable(df, x).values.reshape(-1, 1)
        for degree in degrees:
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)
            self.plot_regression_with_poly(
                df=df, x=x, y=y, X_poly=X_poly, model_class=LinearRegression, 
                model_params_list=[{}], title_suffix=f'Polynomial (degree={degree})', 
                degree=degree
            )

    def polynomial_ridge_regression(
        self,
        df: pd.DataFrame,
        x: str,
        y: list,
        degrees: list,
        alphas: list
    ) -> None:
        """
        Realiza y grafica una regresión polinomial Ridge.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (list): Lista de nombres de columnas para las variables dependientes.
            degrees (list): Lista de grados para la regresión polinomial.
            alphas (list): Lista de valores alpha para la regresión Ridge.
        """
        X = transform_variable(df, x).values.reshape(-1, 1)
        for degree in degrees:
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)
            model_params_list = [{'alpha': alpha} for alpha in alphas]
            self.plot_regression_with_poly(
                df=df, x=x, y=y, X_poly=X_poly, model_class=Ridge, 
                model_params_list=model_params_list, 
                title_suffix=f'Polynomial Ridge (degree={degree})', 
                degree=degree
            )

    def polynomial_lasso_regression(
        self,
        df: pd.DataFrame,
        x: str,
        y: list,
        degrees: list,
        alphas: list
    ) -> None:
        """
        Realiza y grafica una regresión polinomial Lasso.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (list): Lista de nombres de columnas para las variables dependientes.
            degrees (list): Lista de grados para la regresión polinomial.
            alphas (list): Lista de valores alpha para la regresión Lasso.
        """
        X = transform_variable(df, x).values.reshape(-1, 1)
        for degree in degrees:
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)
            model_params_list = [{'alpha': alpha} for alpha in alphas]
            self.plot_regression_with_poly(
                df=df, x=x, y=y, X_poly=X_poly, model_class=Lasso, 
                model_params_list=model_params_list, 
                title_suffix=f'Polynomial Lasso (degree={degree})', 
                degree=degree
            )
