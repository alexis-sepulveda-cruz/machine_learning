"""
Este módulo implementa una clase para la clasificación de datos usando
diferentes modelos de aprendizaje automático. Incluye preprocesamiento de datos,
entrenamiento de modelos, evaluación y visualización de resultados.
"""

import os

from sklearn.decomposition import PCA
from pia.util_csv_reader import UtilCSVReader
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import seaborn as sns


class Classification:
    """
    Clase para realizar clasificación usando varios modelos de aprendizaje
    automático.
    """
    csv_reader: UtilCSVReader = UtilCSVReader()

    def __init__(self, base_path: str = 'img/classification') -> None:
        """
        Inicializa la clase Classification.

        Args:
            base_path (str): Ruta base para guardar las imágenes generadas.
        """
        self.df_train: pd.DataFrame = self.csv_reader.read_csv_to_dataframe(
            'train.csv'
        )
        self.df_test: pd.DataFrame = self.csv_reader.read_csv_to_dataframe(
            'test.csv'
        )
        self.X_train: np.ndarray
        self.X_val: np.ndarray
        self.X_test: np.ndarray
        self.y_train: np.ndarray
        self.y_val: np.ndarray
        self.y_test: np.ndarray
        self.models: Dict[str, BaseEstimator] = {
            'KNN': KNeighborsClassifier(),
            'Logistic Regression': LogisticRegression(),
            'SVM': SVC(probability=True),
            'Decision Tree': DecisionTreeClassifier()
        }
        self.base_path: str = base_path
        self.ensure_directory_exists(self.base_path)

    def ensure_directory_exists(self, base_path: str) -> None:
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

    def preprocess_data(self) -> None:
        """
        Preprocesa los datos, incluyendo la limpieza, división en conjuntos
        de entrenamiento, validación y prueba, y escalado de características.
        """

        # Limpiar y transformar los datos
        self.df_train = self.df_train.dropna()
        self.df_test = self.df_test.dropna()

        # Dividir los datos de entrenamiento en características (X) y etiquetas (y)
        X_train: pd.DataFrame = self.df_train.drop('engagement', axis=1)
        y_train: pd.Series = self.df_train['engagement']

        # Dividir los datos de entrenamiento en conjuntos de entrenamiento y validación
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Preparar datos de prueba
        self.X_test: pd.DataFrame = self.df_test.copy()  # No dividimos df_test, lo usamos completo

        # Escalar las características
        scaler: StandardScaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)

        # Nota: No tenemos y_test ya que df_test no incluye la columna 'engagement'
        self.y_test = None

    def plot_classification_results(self, models: Dict[str, BaseEstimator]) -> None:
        """
        Genera y guarda gráficos de dispersión para visualizar los resultados
        de clasificación de cada modelo.

        Args:
            models (Dict[str, BaseEstimator]): Modelos entrenados.
        """
        # Aplicar PCA para reducir a 2 dimensiones
        pca = PCA(n_components=2)
        X_val_2d = pca.fit_transform(self.X_val)

        # Configurar el estilo de seaborn
        sns.set(style="whitegrid")

        # Crear un gráfico para cada modelo
        for name, model in models.items():
            plt.figure(figsize=(10, 8))
            
            # Obtener predicciones
            y_pred = model.predict(self.X_val)
            
            # Crear el gráfico de dispersión
            scatter = plt.scatter(X_val_2d[:, 0], X_val_2d[:, 1], 
                                  c=y_pred, cmap='coolwarm', alpha=0.6)
            
            # Añadir una leyenda
            plt.colorbar(scatter)
            
            # Configurar etiquetas y título
            plt.xlabel('Primera Componente Principal')
            plt.ylabel('Segunda Componente Principal')
            plt.title(f'Resultados de Clasificación - {name}')
            
            # Guardar la imagen
            img_path: str = os.path.join(self.base_path, f'classification_{name}.png')
            plt.savefig(img_path)
            print(f"Imagen de clasificación para {name} guardada en: {img_path}")
            
            plt.close()

    def plot_results_table(self, results: Dict[str, Dict[str, Any]], 
                       use_grid_search: bool) -> None:
        """
        Genera y guarda una tabla visual con los resultados de cada modelo.

        Args:
            results (Dict[str, Dict[str, Any]]): Resultados de la evaluación.
            use_grid_search (bool): Indica si se usó Grid Search.
        """
        models = list(results.keys())
        metrics = ['CV Score', 'Validation Score', 'Pos. Pred. Test', 'Mean Prob. Test']

        cell_text = []
        for model in models:
            row = [
                f"{results[model]['CV Score']:.4f}",
                f"{results[model]['Validation Score']:.4f}",
                f"{sum(results[model]['Test Predictions'])}",
                f"{np.mean(results[model]['Test Probabilities']):.4f}"
            ]
            cell_text.append(row)

        fig, ax = plt.subplots(figsize=(12, len(models) + 1))
        ax.axis('off')
        table = ax.table(cellText=cell_text,
                        rowLabels=models,
                        colLabels=metrics,
                        cellLoc='center',
                        loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        title = ('Resultados de Clasificación' + 
                (' con Grid Search' if use_grid_search else ' sin Grid Search'))
        plt.title(title)

        # Guardar la imagen en el directorio especificado
        grid_search_suffix = "_grid_search" if use_grid_search else ""
        img_path: str = os.path.join(self.base_path, 
                                    f'results_table{grid_search_suffix}.png')
        plt.savefig(img_path, bbox_inches='tight', dpi=300)
        print(f"Tabla de resultados {'con' if use_grid_search else 'sin'} "
            f"Grid Search guardada en: {img_path}")

        plt.close()

    def train_and_evaluate(
        self, use_grid_search: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Entrena y evalúa los modelos de clasificación.

        Args:
            use_grid_search (bool): Si se debe usar búsqueda en cuadrícula.

        Returns:
            Dict[str, Dict[str, float]]: Resultados de la evaluación.
        """
        results: Dict[str, Dict[str, float]] = {}
        best_models: Dict[str, BaseEstimator] = {}

        for name, model in self.models.items():
            if use_grid_search:
                param_grid: Dict[str, Any] = self.get_param_grid(name)
                grid_search: GridSearchCV = GridSearchCV(
                    model, param_grid, cv=5, scoring='roc_auc'
                )
                grid_search.fit(self.X_train, self.y_train)
                best_model: BaseEstimator = grid_search.best_estimator_
            else:
                best_model = model.fit(self.X_train, self.y_train)

            best_models[name] = best_model

            # Validación cruzada
            cv_score: float = cross_val_score(
                best_model, self.X_train, self.y_train, cv=5, scoring='roc_auc'
            ).mean()

            # Evaluación en conjunto de validación
            val_score: float = roc_auc_score(
                self.y_val, best_model.predict_proba(self.X_val)[:, 1]
            )

            # Predicciones en conjunto de prueba
            test_predictions: np.ndarray = best_model.predict(self.X_test)
            test_proba: np.ndarray = best_model.predict_proba(self.X_test)[:, 1]

            results[name] = {
                'CV Score': cv_score,
                'Validation Score': val_score,
                'Test Predictions': test_predictions,
                'Test Probabilities': test_proba
            }

        self.plot_roc_curves(best_models, use_grid_search)
        self.plot_classification_results(best_models)

        return results

    def get_param_grid(self, model_name: str) -> Dict[str, Any]:
        """
        Obtiene la cuadrícula de parámetros para la búsqueda en cuadrícula.

        Args:
            model_name (str): Nombre del modelo.

        Returns:
            Dict[str, Any]: Cuadrícula de parámetros.
        """
        if model_name == 'KNN':
            return {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        elif model_name == 'Logistic Regression':
            return {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2']
            }
        elif model_name == 'SVM':
            return {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear']
            }
        elif model_name == 'Decision Tree':
            return {
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5, 10]
            }
        else:
            return {}

    def run_classification(self) -> None:
        """
        Ejecuta el proceso de clasificación completo, incluyendo
        preprocesamiento, entrenamiento y evaluación.
        """
        self.preprocess_data()

        print("Resultados sin Grid Search:")
        results_without_gs: Dict[str, Dict[str, float]] = self.train_and_evaluate(
            use_grid_search=False
        )
        self.print_results(results_without_gs)
        self.plot_results_table(results_without_gs, use_grid_search=False)

        print("\nResultados con Grid Search:")
        results_with_gs: Dict[str, Dict[str, float]] = self.train_and_evaluate(
            use_grid_search=True
        )
        self.print_results(results_with_gs)
        self.plot_results_table(results_with_gs, use_grid_search=True)

    def print_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Imprime los resultados de la evaluación de los modelos.

        Args:
            results (Dict[str, Dict[str, Any]]): Resultados a imprimir.
        """
        for model, scores in results.items():
            print(f"\n{model}:")
            for metric, score in scores.items():
                if metric in ['CV Score', 'Validation Score']:
                    print(f"  {metric}: {score:.4f}")
                elif metric == 'Test Predictions':
                    print(f"  Número de predicciones positivas en test: {sum(score)}")
                elif metric == 'Test Probabilities':
                    print(f"  Probabilidad media de clase positiva en test: {np.mean(score):.4f}")

    def plot_roc_curves(self, models: Dict[str, BaseEstimator],
                    use_grid_search: bool) -> None:
        """
        Genera y guarda un gráfico de curvas ROC para los modelos entrenados usando el conjunto de validación.

        Args:
            models (Dict[str, BaseEstimator]): Modelos entrenados.
            use_grid_search (bool): Indica si se usó Grid Search.
        """
        plt.figure(figsize=(10, 8))
        colors: List[str] = ['blue', 'red', 'green', 'orange']

        for (name, model), color in zip(models.items(), colors):
            y_pred_proba: np.ndarray = model.predict_proba(self.X_val)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_val, y_pred_proba)
            auc: float = roc_auc_score(self.y_val, y_pred_proba)
            plt.plot(fpr, tpr, color=color,
                    label=f'{name} (AUC = {auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve - Validation Set')
        plt.legend(loc="lower right")

        # Guardar la imagen en el directorio especificado
        grid_search_suffix = "_grid_search" if use_grid_search else ""
        img_path: str = os.path.join(self.base_path,
                                    f'roc_curve{grid_search_suffix}.png')
        plt.savefig(img_path)
        print(f"Imagen ROC {'con' if use_grid_search else 'sin'} "
            f"Grid Search guardada en: {img_path}")

if __name__ == "__main__":
    classification: Classification = Classification()
    classification.run_classification()