"""
Este módulo proporciona una clase para realizar clasificación 
de activos utilizando varios algoritmos de aprendizaje automático.

Incluye métodos para preprocesar datos, entrenar y evaluar modelos, 
y visualizar resultados.

Dependencias:
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
"""

import pandas as pd
import numpy as np
import os
from typing import Any, List, Tuple, Dict
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score,
    precision_recall_curve, 
    auc,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from utils.score_classification import ScoreClassification


class AssetClassification:
    """
    Clase para realizar clasificación de activos utilizando diversos 
    algoritmos de aprendizaje automático.

    Atributos:
        data (pd.DataFrame): DataFrame que contiene los datos a 
                             clasificar.
        scaler (StandardScaler): Objeto para escalar los datos 
                                 de entrada.
        base_path (str): Ruta base para guardar las imágenes
                         generadas.
    """

    def __init__(
        self, 
        data: pd.DataFrame, 
        base_path: str = 'img/classification'
    ):
        """
        Inicializa la clase AssetClassification.

        Args:
            data (pd.DataFrame): DataFrame que contiene los datos 
                                 a clasificar.
            base_path (str, opcional): Ruta base para guardar las 
                                       imágenes generadas. Por defecto 
                                       es 'img/classification'.
        """
        self.data = data
        self.scaler = StandardScaler()
        self.base_path = base_path
        self.ensure_directory_exists(self.base_path)

    def ensure_directory_exists(
        self, 
        base_path: str
    ) -> None:
        """
        Asegura que el directorio especificado exista, creándolo 
        si es necesario.

        Args:
            base_path (str): Ruta del directorio a verificar/crear.
        """
        if not os.path.exists(base_path):
            try:
                os.makedirs(base_path)
                print(f"Directorio creado: {base_path}")
            except OSError as e:
                print(f"Error al crear el directorio {base_path}: {e}")
        else:
            print(f"El directorio ya existe: {base_path}")

    def preprocess_data(
        self, 
        x: List[str], 
        y: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocesa los datos para el entrenamiento y prueba del modelo.

        Args:
            x (List[str]): Lista de nombres de columnas para las 
                           características.
            y (str): Nombre de la columna objetivo.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                X_train, X_test, y_train, y_test
        """
        X = self.data[x]
        y = self.data[y]
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )

    def evaluate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calcula varias métricas de evaluación para el modelo.

        Args:
            y_true (np.ndarray): Etiquetas verdaderas.
            y_pred (np.ndarray): Predicciones del modelo.
            y_pred_proba (np.ndarray, opcional): Probabilidades de 
                                                 predicción para ROC AUC.

        Returns:
            Dict[str, float]: Diccionario con las métricas calculadas.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(
                y_true, y_pred, average='weighted'
            ),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(
                y_true, y_pred_proba, average='weighted', 
                multi_class='ovr'
            )

        return metrics

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_score: np.ndarray, model_name: str, y: str) -> None:
        """
        Genera y guarda un gráfico de la curva de precisión-recall.

        Args:
            y_true (np.ndarray): Etiquetas verdaderas.
            y_score (np.ndarray): Puntuaciones de predicción del modelo.
            model_name (str): Nombre del modelo utilizado.
            y (str): Nombre de la variable objetivo.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='b', label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Curva de Precisión-Recall para {model_name} - {y}')
        plt.legend(loc="lower left")
        plt.savefig(f'{self.base_path}/pr_curve_{model_name.lower().replace(" ", "_")}_{y}.png')
        plt.show()
        plt.close()

    def plot_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray, model_name: str, y: str) -> None:
        """
        Genera y guarda un gráfico de la curva ROC.

        Args:
            y_true (np.ndarray): Etiquetas verdaderas.
            y_score (np.ndarray): Puntuaciones de predicción del modelo.
            model_name (str): Nombre del modelo utilizado.
            y (str): Nombre de la variable objetivo.
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title(f'Curva ROC para {model_name} - {y}')
        plt.legend(loc="lower right")
        plt.savefig(f'{self.base_path}/roc_curve_{model_name.lower().replace(" ", "_")}_{y}.png')
        plt.show()
        plt.close()

    def perform_grid_search(
        self, 
        model: Any, 
        param_grid: Dict[str, List[Any]], 
        X: np.ndarray, 
        y: np.ndarray, 
        cv: int = 5,
        score_name: str = ScoreClassification.F1_WEIGHTED
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
                              Por defecto, se utiliza 'f1_weighted', que corresponde al valor F1 ponderado. 
                              Otras opciones válidas incluyen 'accuracy', 'precision', 'recall', 
                              'roc_auc', etc.

        Returns:
            Tuple[Any, Dict[str, Any]]: El mejor modelo y los mejores parámetros encontrados.
        """
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=score_name, n_jobs=-1)
        grid_search.fit(X, y)
        
        print("Mejores parámetros encontrados:")
        print(grid_search.best_params_)
        print(f"Mejor puntuación: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_

    def train_and_evaluate(
            self, 
            model, 
            X_train: np.ndarray, 
            X_test: np.ndarray, 
            y_train: np.ndarray, 
            y_test: np.ndarray, 
            model_name: str
    ) -> Tuple[float, float, np.ndarray, Dict[str, float]]:
        """
        Entrena y evalúa un modelo de clasificación.

        Args:
            model: Modelo de clasificación a entrenar y evaluar.
            X_train (np.ndarray): Datos de entrenamiento.
            X_test (np.ndarray): Datos de prueba.
            y_train (np.ndarray): Etiquetas de entrenamiento.
            y_test (np.ndarray): Etiquetas de prueba.
            model_name (str): Nombre del modelo para el reporte.

        Returns:
            Tuple[float, float, np.ndarray, Dict[str, float]]: 
                - Puntuación de entrenamiento, 
                - puntuación de prueba, 
                - predicciones, 
                - métricas
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(
            f"{model_name} - Puntuaciones: "
            f"[entrenamiento {train_score:.4f}, "
            f"prueba {test_score:.4f}]"
        )
        print(classification_report(y_test, y_pred))
        
        y_pred_proba = (
            model.predict_proba(X_test)
            if hasattr(model, "predict_proba")
            else None
        )

        metrics = self.evaluate_metrics(y_test, y_pred, y_pred_proba)
        
        print("Métricas adicionales:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nMatriz de confusión:")
        print(confusion_matrix(y_test, y_pred))

        # Validación cruzada
        cv_results = cross_validate(model, X_train, y_train, cv=5, 
                                    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
        
        print("\nResultados de validación cruzada:")
        for metric, scores in cv_results.items():
            if metric.startswith('test_'):
                print(f"{metric}: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
                
        # Generar y guardar el gráfico de la matriz de confusión
        self.plot_confusion_matrix(y_test, y_pred, model_name, y_train.name)
        if model == LogisticRegression and y_pred_proba is not None:
            self.plot_precision_recall_curve(y_test, y_pred_proba[:, 1], model_name, y_train.name)
            self.plot_roc_curve(y_test, y_pred_proba[:, 1], model_name, y_train.name)
        
        return train_score, test_score, y_pred, metrics

    def plot_results(
            self, 
            X_test: np.ndarray, 
            y_test: np.ndarray, 
            y_pred: np.ndarray, 
            x: List[str], 
            y: str, 
            model_name: str, 
            train_score: float, 
            test_score: float, 
            metrics: Dict[str, float]
    ) -> None:
        """
        Genera y guarda un gráfico de dispersión de los resultados 
        del modelo.

        Args:
            X_test (np.ndarray): Datos de prueba.
            y_test (np.ndarray): Etiquetas reales de prueba.
            y_pred (np.ndarray): Predicciones del modelo.
            x (List[str]): Nombres de las características utilizadas.
            y (str): Nombre de la variable objetivo.
            model_name (str): Nombre del modelo utilizado.
            train_score (float): Puntuación de entrenamiento.
            test_score (float): Puntuación de prueba.
            metrics (Dict[str, float]): Métricas adicionales del modelo.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        scatter = ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis',
            alpha=0.7
        )
        
        incorrect = y_test != y_pred
        ax.scatter(
            X_test[incorrect, 0], X_test[incorrect, 1], c='red', marker='x', 
            s=100, label='Incorrectas'
        )
        
        ax.set_xlabel(x[0])
        ax.set_ylabel(x[1])
        ax.set_title(
            f"{model_name} - Puntuaciones: "
            f"[entrenamiento {train_score:.4f}, "
            f"prueba {test_score:.4f}]"
        )
        
        # Agregar métricas al gráfico
        metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        ax.text(
            0.05, 0.95, metrics_text, transform=ax.transAxes, 
            verticalalignment='top',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.8
            )
        )
        ax.legend()
        
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig(
            f'{self.base_path}/{model_name.lower().replace(" ", "_")}'
            f'_{y}_{"_".join(x)}.png'
        )
        plt.show()
        plt.close()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        y: str
    ) -> None:

        """
        Genera y guarda un gráfico de la matriz de confusión.

        Args:
            y_true (np.ndarray): Etiquetas verdaderas.
            y_pred (np.ndarray): Predicciones del modelo.
            model_name (str): Nombre del modelo utilizado.
            y (str): Nombre de la variable objetivo.
        """
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, ax=ax
        )
        
        ax.set_title(f'Matriz de Confusión para {model_name} - {y}')
        ax.set_ylabel('Etiqueta Verdadera')
        ax.set_xlabel('Etiqueta Predicha')
        
        plt.tight_layout()
        plt.savefig(
            f'{self.base_path}/confusion_matrix_{model_name.lower().replace(" ", "_")}'
            f'_{y}.png'
        )
        plt.show()
        plt.close()

    def knn_classifier(
            self,
            x: List[str],
            y: str,
            n_neighbors: int = 0
    ) -> None:
        """
        Entrena y evalúa un clasificador K-Nearest Neighbors.

        Args:
            x (List[str]): Lista de nombres de columnas para las
                           características.
            y (str): Nombre de la columna objetivo.
            n_neighbors (int, opcional): Número de vecinos a considerar.
                                         Por defecto es 5.
        """
        X_train, X_test, y_train, y_test = self.preprocess_data(x, y)

        if n_neighbors > 0:
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model_name = "KNN"
        else:
            param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
            base_model = KNeighborsClassifier()
            model = self.perform_grid_search(base_model, param_grid, X_train, y_train)
            model_name = "KNN (optimizado)"

        train_score, test_score, y_pred, metrics = self.train_and_evaluate(
            model, X_train, X_test, y_train, y_test, model_name
        )

        self.plot_results(
            X_test, y_test, y_pred, x, y, model_name,
            train_score, test_score, metrics
        )

    def logistic_regression(
            self, 
            x: List[str], 
            y: str, 
            random_state: int = 0
    ) -> None:
        """
        Entrena y evalúa un modelo de regresión logística.

        Args:
            x (List[str]): Lista de nombres de columnas para las 
                           características.
            y (str): Nombre de la columna objetivo.
        """
        X_train, X_test, y_train, y_test = self.preprocess_data(x, y)

        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        base_model = LogisticRegression(random_state=random_state)
        model, best_params = self.perform_grid_search(base_model, param_grid, X_train, y_train)
        model_name = "Regresión Logística (optimizada)"

        train_score, test_score, y_pred, metrics = self.train_and_evaluate(
            model, X_train, X_test, y_train, y_test, model_name
        )
        self.plot_results(
            X_test, y_test, y_pred, x, y, model_name,
            train_score, test_score, metrics
        )

    def svm_classifier(
            self, 
            x: List[str], 
            y: str, 
            random_state: int = 0
    ) -> None:
        """
        Entrena y evalúa un clasificador Support Vector Machine.

        Args:
            x (List[str]): Lista de nombres de columnas para las características.
            y (str): Nombre de la columna objetivo.
        """
        X_train, X_test, y_train, y_test = self.preprocess_data(x, y)
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.1, 1]
        }
        base_model = SVC(random_state=random_state, probability=True)
        model, best_params = self.perform_grid_search(base_model, param_grid, X_train, y_train)
        model_name = "SVM (optimizado)"

        train_score, test_score, y_pred, metrics = self.train_and_evaluate(
            model, X_train, X_test, y_train, y_test, model_name
        )
        self.plot_results(
            X_test, y_test, y_pred, x, y, model_name, 
            train_score, test_score, metrics
        )

    def decision_tree(
            self, 
            x: List[str], 
            y: str, 
            random_state: int = 0
    ) -> None:
        """
        Entrena y evalúa un clasificador de Árbol de Decisión.

        Args:
            x (List[str]): Lista de nombres de columnas para las características.
            y (str): Nombre de la columna objetivo.
        """
        X_train, X_test, y_train, y_test = self.preprocess_data(x, y)
        param_grid = {
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        base_model = DecisionTreeClassifier(random_state=random_state)
        model, best_params = self.perform_grid_search(base_model, param_grid, X_train, y_train)
        model_name = "Árbol de Decisión (optimizado)"
        
        train_score, test_score, y_pred, metrics = self.train_and_evaluate(
            model, X_train, X_test, y_train, y_test, model_name
        )
        self.plot_results(
            X_test, y_test, y_pred, x, y, model_name,
            train_score, test_score, metrics
        )
