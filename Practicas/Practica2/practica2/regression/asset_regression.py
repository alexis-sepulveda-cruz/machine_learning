"""
Este módulo proporciona la clase AssetRegression para 
realizar análisis de regresión en datos de activos.

El módulo incluye funcionalidades para entrenar y evaluar 
varios modelos de regresión, incluyendo regresión lineal, 
Ridge, Lasso, polinomial, KNN y árboles de decisión.

También incluye métodos para visualizar los resultados de la 
regresión.

Clases:
    AssetRegression: Clase principal para realizar análisis de 
    regresión en datos de activos.

Dependencias:
    - pandas
    - numpy
    - matplotlib
    - scikit-learn
    - practica2.utils.util
"""

import os
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from practica2.utils.score_regression import ScoreRegression
from practica2.utils.util import ensure_directory_exists, transform_variable
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split


class AssetRegression:
    """
    Clase para realizar análisis de regresión en datos de activos.

    Esta clase proporciona métodos para entrenar y evaluar varios 
    modelos de regresión, así como para visualizar los resultados.

    Atributos:
        data (pd.DataFrame): Los datos de activos.
        scaler (StandardScaler): Escalador para normalizar los datos.
        base_path (str): Ruta base para guardar las imágenes generadas.
    """

    def __init__(self, data: pd.DataFrame, base_path: str = 'img/regression'):
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
    
    def perform_grid_search(
        self, 
        model: Any, 
        param_grid: Dict[str, List[Any]], 
        X: np.ndarray, 
        y: np.ndarray, 
        cv: int = 5,
        score_name: str = ScoreRegression.R2
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Realiza una búsqueda de cuadrícula para encontrar los mejores hiperparámetros para el modelo.

        Args:
            model (Any): El modelo base para realizar la búsqueda de cuadrícula.
            param_grid (Dict[str, List[Any]]): Diccionario de parámetros para buscar.
            X (np.ndarray): Datos de características.
            y (np.ndarray): Etiquetas objetivo.
            cv (int): Número de pliegues para la validación cruzada.
            score_name (str): Nombre de la métrica de rendimiento que se utilizará para evaluar los
                              diferentes hiperparámetros del modelo durante la búsqueda de cuadrícula. 
                              Por defecto, se utiliza 'r2', que corresponde 

        Returns:
            Tuple[Any, Dict[str, Any]]: El mejor modelo y los mejores parámetros encontrados.
        """
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=score_name, n_jobs=-1)
        grid_search.fit(X, y)
        
        print("Mejores parámetros encontrados:")
        print(grid_search.best_params_)
        print(f"Mejor puntuación: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_class: type,
        model_params: dict
    ) -> object:
        """
        Entrena un modelo de regresión.

        Args:
            X (np.ndarray): Variables independientes.
            y (np.ndarray): Variable dependiente.
            model_class (type): Clase del modelo a entrenar.
            model_params (dict): Parámetros para el modelo.

        Returns:
            object: El modelo entrenado.
        """
        model = model_class(**model_params)
        model.fit(X, y)
        return model

    def plot_regression(
        self,
        df: pd.DataFrame,
        x: str,
        y_columns: list,
        model_class: type,
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

    def linear_regression(self, df: pd.DataFrame, x: str, y: list) -> None:
        """
        Realiza y grafica una regresión lineal.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (list): Lista de nombres de columnas para las variables 
                      dependientes.
        """
        self.plot_regression(df, x, y, LinearRegression, [{}], 'Linear')

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
        self.plot_regression(df, x, y, Ridge, model_params_list, 'Ridge')

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
        self.plot_regression(df, x, y, Lasso, model_params_list, 'Lasso')

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
                df, x, y, X_poly, LinearRegression, [{}], 
                f'Polynomial (degree={degree})', degree
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
                df, x, y, X_poly, Ridge, model_params_list, 
                f'Polynomial Ridge (degree={degree})', degree
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
                df, x, y, X_poly, Lasso, model_params_list, 
                f'Polynomial Lasso (degree={degree})', degree
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
        self.plot_regression(df, x, y, KNeighborsRegressor, model_params_list, 'KNN')

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
            df, x, y, DecisionTreeRegressor, model_params_list, 'Decision Tree'
        )

    def plot_regression_with_poly(
        self,
        df: pd.DataFrame,
        x: str,
        y_columns: list,
        X_poly: np.ndarray,
        model_class: type,
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

    def train_and_evaluate_model(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        model_class: type = LinearRegression,
        model_params: dict = {},
        cv: int = 5,
        perform_grid_search: bool = False, 
        param_grid: Dict[str, List[Any]] = None
    ) -> None:
        """
        Entrena y evalúa un modelo de regresión usando validación cruzada.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (str): Nombre de la columna para la variable dependiente.
            model_class (type, optional): Clase del modelo a utilizar.
                                          Por defecto es LinearRegression.
            model_params (dict, optional): Parámetros para el modelo.
                                           Por defecto es un diccionario vacío.
            cv (int, optional): Número de pliegues para la validación cruzada.
                                Por defecto es 5.
            perform_grid_search (bool, optional): Si se debe realizar una búsqueda 
                                                  de cuadrícula. Por defecto es False.
            param_grid (Dict[str, List[Any]], optional): Diccionario de parámetros para 
                                                         la búsqueda de cuadrícula.
        """
        X = transform_variable(df, x).values.reshape(-1, 1)
        y_data = df[y]

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.2, random_state=42)

        # Entrenar el modelo
        if perform_grid_search and param_grid is not None:
            model = model_class(**model_params)
            best_model, best_params = self.perform_grid_search(model, param_grid, X_train, y_train)
            print(f"Mejores parámetros encontrados: {best_params}")
            model = best_model
        else:
            model = self.train_model(X_train, y_train, model_class, model_params)

        # Realizar predicciones
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Entrenar el modelo con validación cruzada
        scoring = {
            'mse': 'neg_mean_squared_error',
            'rmse': 'neg_root_mean_squared_error',
            'r2': 'r2'
        }
        cv_results = cross_validate(model_class(**model_params), X, y_data, cv=cv, scoring=scoring, return_train_score=True)

        # Promediar los resultados de la validación cruzada
        train_mse = -cv_results['train_mse'].mean()
        train_rmse = np.sqrt(-cv_results['train_rmse'].mean())
        train_r2 = cv_results['train_r2'].mean()
        test_mse = -cv_results['test_mse'].mean()
        test_rmse = np.sqrt(-cv_results['test_rmse'].mean())
        test_r2 = cv_results['test_r2'].mean()

        print(f"Modelo: {model_class.__name__} con parámetros {model_params}")
        print(
            f"Entrenamiento - MSE: {train_mse}, RMSE: {train_rmse}, R2: {train_r2}"
        )
        print(
            f"Validación cruzada - MSE: {test_mse}, RMSE: {test_rmse}, R2: {test_r2}"
        )

        # Imprimir las métricas
        print(f'Train RMSE: {train_rmse}')
        print(f'Test RMSE: {test_rmse}')
        print(f'Train R2: {train_r2}')
        print(f'Test R2: {test_r2}')

        # Graficar resultados
        self._plot_train_test_results(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, x, y, model_class.__name__)

    def train_and_evaluate_model_with_poly(
        self,
        df: pd.DataFrame,
        x: str,
        y_columns: list,
        degree: int,
        model_class: type = LinearRegression,
        model_params_list: list = [{}],
        cv: int = 5,
        perform_grid_search: bool = False, 
        param_grid: Dict[str, List[Any]] = None
    ) -> None:
        """
        Entrena y evalúa un modelo de regresión polinomial usando validación cruzada.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y_columns (list): Lista de nombres de columnas para las variables 
                              dependientes.
            degree (int): Grado para la regresión polinomial.
            model_class (type, optional): Clase del modelo a utilizar.
                                          Por defecto es LinearRegression.
            model_params_list (list, optional): Lista de diccionarios con parámetros 
                                                para el modelo. Por defecto es una lista con un diccionario vacío.
            cv (int, optional): Número de pliegues para la validación cruzada.
                                Por defecto es 5.
            perform_grid_search (bool, optional): Si se debe realizar una búsqueda 
                                                  de cuadrícula. Por defecto es False.
            param_grid (Dict[str, List[Any]], optional): Diccionario de parámetros para 
                                                         la búsqueda de cuadrícula.
        """
        X = transform_variable(df, x).values.reshape(-1, 1)
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        
        for y in y_columns:
            y_data = df[y]

            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(
                X_poly, y_data, test_size=0.2, random_state=42
            )

            for params in model_params_list:
                # Entrenar el modelo
                if perform_grid_search and param_grid is not None:
                    model = model_class(**params)
                    best_model, best_params = self.perform_grid_search(
                        model, param_grid, X_train, y_train
                    )
                    print(f"Mejores parámetros encontrados: {best_params}")
                    model = best_model
                else:
                    model = self.train_model(X_train, y_train, model_class, params)

                # Realizar predicciones
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Entrenar el modelo con validación cruzada
                scoring = {
                    'mse': 'neg_mean_squared_error',
                    'rmse': 'neg_root_mean_squared_error',
                    'r2': 'r2'
                }
                cv_results = cross_validate(
                    model_class(**params), X_poly, y_data, cv=cv, 
                    scoring=scoring, return_train_score=True
                )

                # Promediar los resultados de la validación cruzada
                train_mse = -cv_results['train_mse'].mean()
                train_rmse = np.sqrt(-cv_results['train_rmse'].mean())
                train_r2 = cv_results['train_r2'].mean()
                test_mse = -cv_results['test_mse'].mean()
                test_rmse = np.sqrt(-cv_results['test_rmse'].mean())
                test_r2 = cv_results['test_r2'].mean()

                print(f"Modelo: {model_class.__name__} con parámetros {params}")
                print(
                    f"Entrenamiento - MSE: {train_mse}, RMSE: {train_rmse}, R2: {train_r2}"
                )
                print(
                    f"Validación cruzada - MSE: {test_mse}, RMSE: {test_rmse}, R2: {test_r2}"
                )

                # Imprimir las métricas
                print(f'Train RMSE: {train_rmse}')
                print(f'Test RMSE: {test_rmse}')
                print(f'Train R2: {train_r2}')
                print(f'Test R2: {test_r2}')

                # Graficar resultados
                self._plot_train_test_results(
                    X_train[:, 1], X_test[:, 1], y_train, y_test, y_train_pred, 
                    y_test_pred, x, y, f"{model_class.__name__} {params}"
                )
    

    def _plot_train_test_results(self, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, x, y, model_name):
        """
        Grafica los resultados del entrenamiento y prueba.

        Args:
            X_train (np.ndarray): Datos de entrenamiento X.
            X_test (np.ndarray): Datos de prueba X.
            y_train (np.ndarray): Datos de entrenamiento y.
            y_test (np.ndarray): Datos de prueba y.
            y_train_pred (np.ndarray): Predicciones de entrenamiento.
            y_test_pred (np.ndarray): Predicciones de prueba.
            x (str): Nombre de la variable independiente.
            y (str): Nombre de la variable dependiente.
            model_name (str): Nombre del modelo.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(X_train, y_train, color='blue', label='Train data')
        ax.scatter(X_test, y_test, color='green', label='Test data')
        ax.plot(X_train, y_train_pred, color='black', label='Train fit')
        ax.plot(X_test, y_test_pred, color='red', label='Test fit')

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f'{model_name} Regression: {y} vs {x}')
        ax.legend(loc='upper left')

        save_path = f'{self.base_path}/{y}_vs_{x}_{model_name}_train_test.png'
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close(fig)
        print(f'Gráfico guardado en: {save_path}')

    def train_and_evaluate_linear(
            self, 
            df: pd.DataFrame, 
            x: str, 
            y: str, 
            perform_grid_search: bool = False
    ) -> None:
        """
        Entrena y evalúa un modelo de regresión lineal.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (str): Nombre de la columna para la variable dependiente.
            perform_grid_search (bool, optional): Si se debe realizar una búsqueda de 
                                                  cuadrícula. Por defecto es False.
        """
        param_grid = {
            'fit_intercept': [True, False], 
            'positive': [True, False]
        } if perform_grid_search else None

        self.train_and_evaluate_model(
            df, x, y, model_class=LinearRegression,
            perform_grid_search=perform_grid_search, 
            param_grid=param_grid
        )

    def train_and_evaluate_ridge(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        alpha: float = 1.0, 
        perform_grid_search: bool = False
    ) -> None:
        """
        Entrena y evalúa un modelo de regresión Ridge.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (str): Nombre de la columna para la variable dependiente.
            alpha (float, optional): Parámetro de regularización. Por defecto es 1.0.
            perform_grid_search (bool, optional): Si se debe realizar una búsqueda de 
                                                  cuadrícula. Por defecto es False.
        """
        param_grid = {
            'alpha': [0.1, 1.0, 10.0]
        } if perform_grid_search else None

        self.train_and_evaluate_model(
            df, x, y, model_class=Ridge, model_params={'alpha': alpha}, 
            perform_grid_search=perform_grid_search, param_grid=param_grid
        )

    def train_and_evaluate_lasso(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        alpha: float = 1.0, 
        perform_grid_search: bool = False
    ) -> None:
        """
        Entrena y evalúa un modelo de regresión Lasso.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (str): Nombre de la columna para la variable dependiente.
            alpha (float, optional): Parámetro de regularización. Por defecto es 1.0.
            perform_grid_search (bool, optional): Si se debe realizar una búsqueda de 
                                                  cuadrícula. Por defecto es False.
        """
        param_grid = {
            'alpha': [0.1, 1.0, 10.0]
        } if perform_grid_search else None

        self.train_and_evaluate_model(
            df, x, y, model_class=Lasso, model_params={'alpha': alpha}, 
            perform_grid_search=perform_grid_search, param_grid=param_grid
        )

    def train_and_evaluate_polynomial(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        degree: int = 2, 
        perform_grid_search: bool = False
    ) -> None:
        """
        Entrena y evalúa un modelo de regresión polinomial.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (str): Nombre de la columna para la variable dependiente.
            degree (int, optional): Grado del polinomio. Por defecto es 2.
            perform_grid_search (bool, optional): Si se debe realizar una búsqueda de 
                                                  cuadrícula. Por defecto es False.
        """
        X = transform_variable(df, x).values.reshape(-1, 1)
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        poly_df = pd.DataFrame(X_poly, columns=[f'{x}^{i}' for i in range(degree+1)])
        poly_df[y] = df[y].values

        param_grid = {
            'fit_intercept': [True, False]
        } if perform_grid_search else None
        
        for column in poly_df.columns.difference([y]):
            self.train_and_evaluate_model(
                poly_df, column, y, model_class=LinearRegression, 
                perform_grid_search=perform_grid_search, param_grid=param_grid
            )

    def train_and_evaluate_polynomial_ridge(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        degree: int = 2,
        alpha: float = 1.0, 
        perform_grid_search: bool = False
    ) -> None:
        """
        Entrena y evalúa un modelo de regresión polinomial Ridge.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (str): Nombre de la columna para la variable dependiente.
            degree (int, optional): Grado del polinomio. Por defecto es 2.
            alpha (float, optional): Parámetro de regularización. Por defecto es 1.0.
            perform_grid_search (bool, optional): Si se debe realizar una búsqueda de 
                                                  cuadrícula. Por defecto es False.
        """
        X = transform_variable(df, x).values.reshape(-1, 1)
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        poly_df = pd.DataFrame(X_poly, columns=[f'{x}^{i}' for i in range(degree+1)])
        poly_df[y] = df[y].values

        param_grid = {
            'alpha': [0.1, 1.0, 10.0], 
            'fit_intercept': [True, False]
        } if perform_grid_search else None

        for column in poly_df.columns.difference([y]):
            self.train_and_evaluate_model(
                poly_df, column, y, model_class=Ridge, 
                model_params={'alpha': alpha}, 
                perform_grid_search=perform_grid_search, 
                param_grid=param_grid
            )

    def train_and_evaluate_polynomial_lasso(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        degree: int = 2,
        alpha: float = 1.0, 
        perform_grid_search: bool = False
    ) -> None:

        """
        Entrena y evalúa un modelo de regresión polinomial Lasso.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (str): Nombre de la columna para la variable dependiente.
            degree (int, optional): Grado del polinomio. Por defecto es 2.
            alpha (float, optional): Parámetro de regularización. Por defecto es 1.0.
            perform_grid_search (bool, optional): Si se debe realizar una búsqueda de 
                                                  cuadrícula. Por defecto es False.
        """
        X = transform_variable(df, x).values.reshape(-1, 1)
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        poly_df = pd.DataFrame(X_poly, columns=[f'{x}^{i}' for i in range(degree+1)])
        poly_df[y] = df[y].values

        param_grid = {
            'alpha': [0.1, 1.0, 10.0], 
            'fit_intercept': [True, False]
        } if perform_grid_search else None

        for column in poly_df.columns.difference([y]):
            self.train_and_evaluate_model(
                poly_df, column, y, model_class=Lasso, model_params={'alpha': alpha}, 
                perform_grid_search=perform_grid_search, param_grid=param_grid
            )
    
    def train_and_evaluate_knn(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        n_neighbors: int = 5, 
        perform_grid_search: bool = False
    ) -> None:
        """
        Entrena y evalúa un modelo de regresión KNN.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (str): Nombre de la columna para la variable dependiente.
            n_neighbors (int, optional): Número de vecinos. Por defecto es 5.
            perform_grid_search (bool, optional): Si se debe realizar una búsqueda de 
                                                  cuadrícula. Por defecto es False.
        """
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11]
        } if perform_grid_search else None

        self.train_and_evaluate_model(
            df, x, y, model_class=KNeighborsRegressor, 
            model_params={'n_neighbors': n_neighbors}, 
            perform_grid_search=perform_grid_search, 
            param_grid=param_grid
        )

    def train_and_evaluate_decision_tree(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        max_depth: int = None, 
        perform_grid_search: bool = False
    ) -> None:
        """
        Entrena y evalúa un modelo de regresión con árbol de decisión.

        Args:
            df (pd.DataFrame): Los datos.
            x (str): Nombre de la columna para la variable independiente.
            y (str): Nombre de la columna para la variable dependiente.
            max_depths (int): La profundidad máxima del árbol.
            perform_grid_search (bool, optional): Si se debe realizar una búsqueda de 
                                                  cuadrícula. Por defecto es False.
        """
        param_grid = {
            'max_depth': [None, 5, 10, 15, 20], 
            'min_samples_split': [2, 5, 10]
        } if perform_grid_search else None

        self.train_and_evaluate_model(
            df, x, y, model_class=DecisionTreeRegressor, 
            model_params={'max_depth': max_depth}, 
            perform_grid_search=perform_grid_search, 
            param_grid=param_grid
        )