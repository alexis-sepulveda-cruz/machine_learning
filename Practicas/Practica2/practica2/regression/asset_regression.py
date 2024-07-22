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

from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from practica2.utils.score_regression import ScoreRegression
from practica2.utils.util import ensure_directory_exists, transform_variable
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, train_test_split


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
    
    def _perform_grid_search(
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
    
    def _plot_train_test_results(
            self, 
            X_train: np.ndarray, 
            X_test: np.ndarray, 
            y_train: np.ndarray, 
            y_test: np.ndarray, 
            y_train_pred: np.ndarray, 
            y_test_pred: np.ndarray, 
            x: str, 
            y: str, 
            model_name: str
    ):
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

    def _train_model(
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

    def _train_and_evaluate_model(
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
            best_model, best_params = self._perform_grid_search(model, param_grid, X_train, y_train)
            print(f"Mejores parámetros encontrados: {best_params}")
            model = best_model
        else:
            model = self._train_model(X_train, y_train, model_class, model_params)

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

    def linear_regression(
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

        self._train_and_evaluate_model(
            df, x, y, model_class=LinearRegression,
            perform_grid_search=perform_grid_search, 
            param_grid=param_grid
        )

    def ridge_regression(
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

        self._train_and_evaluate_model(
            df, x, y, model_class=Ridge, model_params={'alpha': alpha}, 
            perform_grid_search=perform_grid_search, param_grid=param_grid
        )

    def lasso_regression(
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

        self._train_and_evaluate_model(
            df, x, y, model_class=Lasso, model_params={'alpha': alpha}, 
            perform_grid_search=perform_grid_search, param_grid=param_grid
        )

    def _plot_nonlinear_results(
        self, 
        X: np.ndarray, 
        y_data: np.ndarray, 
        model: Any, 
        x_col: str, 
        y_col: str, 
        model_name: str
    ):
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_plot = model.predict(X_plot)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(X, y_data, color='blue', label='Datos')
        ax.plot(X_plot, y_plot, color='red', label='Predicciones del modelo')
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'{model_name}: {y_col} vs {x_col}')
        ax.legend(loc='upper left')
        
        save_path = f'{self.base_path}/{y_col}_vs_{x_col}_{model_name}_nonlinear.png'
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close(fig)
        print(f'Gráfico guardado en: {save_path}')

    def knn_regression(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        n_neighbors: int = 6, 
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
        X = transform_variable(df, x).values.reshape(-1, 1)
        y_data = df[y].values

        X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.2, random_state=42)

        param_grid = {
            'n_neighbors': [3, 5, 6]
        } if perform_grid_search else None

        if perform_grid_search and param_grid is not None:
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
            best_model, best_params = self._perform_grid_search(model, param_grid, X_train, y_train)
            print(f"Mejores parámetros encontrados: {best_params}")
            model = best_model
        else:
            model = self._train_model(X_train, y_train, KNeighborsRegressor, {'n_neighbors': n_neighbors})

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calcular métricas
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"Modelo: KNN Regressor con {n_neighbors} vecinos")
        print(f"Entrenamiento - MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
        print(f"Prueba - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

        # Realizar validación cruzada
        cv_scores = cross_val_score(model, X, y_data, cv=5, scoring='r2')
        print(f"Validación cruzada R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Llamar a la función de trazado no lineal
        self._plot_nonlinear_results(X, y_data, model, x, y, "KNN Regressor")

    def decision_tree_regression(
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
            max_depth (int): La profundidad máxima del árbol.
            perform_grid_search (bool, optional): Si se debe realizar una búsqueda de 
                                                cuadrícula. Por defecto es False.
        """
        X = transform_variable(df, x).values.reshape(-1, 1)
        y_data = df[y].values

        X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.2, random_state=42)

        param_grid = {
            'max_depth': [None, 5, 10, 15, 20], 
            'min_samples_split': [2, 5, 10]
        } if perform_grid_search else None

        if perform_grid_search and param_grid is not None:
            model = DecisionTreeRegressor(max_depth=max_depth)
            best_model, best_params = self._perform_grid_search(model, param_grid, X_train, y_train)
            print(f"Mejores parámetros encontrados: {best_params}")
            model = best_model
        else:
            model = self._train_model(X_train, y_train, DecisionTreeRegressor, {'max_depth': max_depth})

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calcular métricas
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"Modelo: Decision Tree Regressor con max_depth={max_depth}")
        print(f"Entrenamiento - MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
        print(f"Prueba - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

        # Realizar validación cruzada
        cv_scores = cross_val_score(model, X, y_data, cv=5, scoring='r2')
        print(f"Validación cruzada R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Llamar a la función de trazado no lineal
        self._plot_nonlinear_results(X, y_data, model, x, y, "Decision Tree Regressor")
    
    def _plot_polynomial_results(
        self, 
        X: np.ndarray, 
        model: Any, 
        poly: PolynomialFeatures, 
        x_col: str, 
        y_col: str, 
        model_name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        y_train_pred: np.ndarray,
        y_test_pred: np.ndarray
    ):
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_plot_poly = poly.transform(X_plot)
        y_plot = model.predict(X_plot_poly)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(X_train[:, 1], y_train, color='blue', label='Datos de entrenamiento', alpha=0.5)
        ax.scatter(X_test[:, 1], y_test, color='green', label='Datos de prueba', alpha=0.5)
        ax.plot(X_plot, y_plot, color='red', label='Ajuste polinomial')
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'{model_name} Regresión Polinomial: {y_col} vs {x_col}')
        ax.legend(loc='upper left')
        
        save_path = f'{self.base_path}/{y_col}_vs_{x_col}_{model_name}_polynomial.png'
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close(fig)
        print(f'Gráfico guardado en: {save_path}')

        # Gráfico de residuos
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(y_train_pred, y_train_pred - y_train, color='blue', label='Entrenamiento', alpha=0.5)
        ax.scatter(y_test_pred, y_test_pred - y_test, color='green', label='Prueba', alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Valores predichos')
        ax.set_ylabel('Residuos')
        ax.set_title(f'Gráfico de Residuos - {model_name} Regresión Polinomial')
        ax.legend(loc='upper left')

        save_path = f'{self.base_path}/{y_col}_vs_{x_col}_{model_name}_polynomial_residuals.png'
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close(fig)
        print(f'Gráfico de residuos guardado en: {save_path}')
    
    def _train_and_evaluate_polynomial_model(
        self,
        df: pd.DataFrame,
        x: str,
        y_col: str,
        degree: int,
        model_class: type = LinearRegression,
        model_params: dict = {},
        cv: int = 5,
        perform_grid_search: bool = False, 
        param_grid: Dict[str, List[Any]] = None
    ) -> None:
        X = transform_variable(df, x).values.reshape(-1, 1)
        y_data = df[y_col].values
        
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y_data, test_size=0.2, random_state=42)
        
        if perform_grid_search and param_grid is not None:
            model = model_class(**model_params)
            best_model, best_params = self._perform_grid_search(model, param_grid, X_train, y_train)
            print(f"Mejores parámetros encontrados: {best_params}")
            model = best_model
        else:
            model = self._train_model(X_train, y_train, model_class, model_params)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calcular métricas para el conjunto de entrenamiento y prueba
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"Modelo: {model_class.__name__} (Polinomial grado {degree}) con parámetros {model_params}")
        print(f"Entrenamiento - MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
        print(f"Prueba - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

        # Realizar validación cruzada
        cv_scores = cross_val_score(model, X_poly, y_data, cv=cv, scoring='r2')
        print(f"Validación cruzada R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Llamar a una función de trazado especializada para polinomios
        self._plot_polynomial_results(
            X, model, poly, x, y_col, model_class.__name__, 
            X_train, X_test, y_train, y_test, y_train_pred, y_test_pred
        )

    def polynomial_regression(
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
        param_grid = {
            'fit_intercept': [True, False]
        } if perform_grid_search else None
        
        self._train_and_evaluate_polynomial_model(
            df, x, y, degree, 
            model_class=LinearRegression, 
            perform_grid_search=perform_grid_search, 
            param_grid=param_grid
        )
    
    def polynomial_ridge_regression(
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
        param_grid = {
            'alpha': [0.1, 1.0, 10.0], 
            'fit_intercept': [True, False]
        } if perform_grid_search else None

        self._train_and_evaluate_polynomial_model(
            df, x, y, degree,
            model_class=Ridge, 
            model_params={'alpha': alpha},
            perform_grid_search=perform_grid_search, 
            param_grid=param_grid
        )

    def polynomial_lasso_regression(
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
        param_grid = {
            'alpha': [0.1, 1.0, 10.0], 
            'fit_intercept': [True, False]
        } if perform_grid_search else None

        self._train_and_evaluate_polynomial_model(
            df, x, y, degree,
            model_class=Lasso, 
            model_params={'alpha': alpha},
            perform_grid_search=perform_grid_search, 
            param_grid=param_grid
        )
