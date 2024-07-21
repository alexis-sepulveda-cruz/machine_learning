import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from kneed import KneeLocator
from practica4.utils.util import ensure_directory_exists
from sklearn.metrics import classification_report


class AssetCluster:
    def __init__(
        self, 
        data: pd.DataFrame, 
        base_path: str = 'img/cluster'
    ):
        """
        Inicializa la clase AssetCluster.

        Args:
            data (pd.DataFrame): DataFrame con los datos a analizar.
            base_path (str): Ruta base para guardar las imágenes generadas.
        """
        self.data = data
        self.scaler = StandardScaler()
        self.base_path = base_path
        ensure_directory_exists(self.base_path)

    def elbow_and_kmeans(self, features: List[str], max_k: int = 10, max_iterations: int = 15) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Aplica el método del codo y luego realiza el agrupamiento K-means.

        Args:
            features (List[str]): Lista de características para el clustering.
            max_k (int): Número máximo de clusters a considerar en el método del codo.
            max_iterations (int): Número máximo de iteraciones para K-means.

        Returns:
            Tuple[np.ndarray, np.ndarray, int]: Centroides, etiquetas de cluster y número óptimo de clusters.
        """
        X = self.scaler.fit_transform(self.data[features])
        
        # Método del codo
        inertias = []
        k_range = range(1, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        # Calcular los incrementos de inercia
        inertia_diffs = np.diff(inertias)

        # Usar KneeLocator para encontrar el "codo" en la gráfica de inercia
        kneedle = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
        optimal_k = kneedle.elbow

        # Graficar el método del codo y el gráfico de Pareto
        self._plot_elbow_and_pareto(k_range, inertias, inertia_diffs, optimal_k)

        print(f'El mejor número de clusters (k) basado en el método del codo es: {optimal_k}')

        # K-means con el número óptimo de clusters
        centroids = X[np.random.choice(X.shape[0], optimal_k, replace=False)]
        
        for iteration in range(max_iterations):
            old_centroids = centroids.copy()
            
            # Asignar puntos a clusters
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Actualizar centroides
            for i in range(optimal_k):
                centroids[i] = X[labels == i].mean(axis=0)

            self._plot_clusters(X, labels, centroids, f"kmeans_clusters_iteration_{iteration}.png")
            
            # Criterio de paro
            if np.array_equal(old_centroids, centroids):
                break

        return centroids, labels, optimal_k

    def _plot_elbow_and_pareto(self, k_range, inertias, inertia_diffs, optimal_k):
        """
        Genera los gráficos del método del codo y de Pareto.

        Args:
            k_range: Rango de valores de k.
            inertias: Lista de inercias.
            inertia_diffs: Diferencias de inercia.
            optimal_k: Número óptimo de clusters.
        """
        plt.figure(figsize=(14, 6))

        # Gráfico del método del codo
        plt.subplot(1, 2, 1)
        plt.plot(k_range, inertias, marker='o')
        plt.axvline(optimal_k, color='r', linestyle='--', label=f'Óptimo k = {optimal_k}')
        plt.xlabel('Número de clusters, k')
        plt.ylabel('Inercia')
        plt.title('Método del codo para determinar el número óptimo de clusters')
        plt.legend()

        # Gráfico de Pareto
        plt.subplot(1, 2, 2)
        plt.bar(range(2, len(k_range) + 1), np.abs(inertia_diffs), color='b', alpha=0.6, label='Incremento de inercia')
        plt.axvline(optimal_k, color='r', linestyle='--', label=f'Óptimo k = {optimal_k}')
        plt.xlabel('Número de clusters, k')
        plt.ylabel('Incremento de inercia')
        plt.title('Gráfico de Pareto de los incrementos de inercia')

        # Añadir línea acumulativa
        cumulative_increments = np.cumsum(np.abs(inertia_diffs))
        plt.plot(range(2, len(k_range) + 1), cumulative_increments, color='r', marker='o', linestyle='dashed', label='Incremento acumulado')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.base_path}/elbow_and_pareto.png")
        plt.show()
        plt.close()

    def _plot_clusters(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray, filename: str):
        """
        Genera un gráfico de dispersión de los clusters.

        Args:
            X (np.ndarray): Datos escalados.
            labels (np.ndarray): Etiquetas de cluster.
            centroids (np.ndarray): Centroides de los clusters.
            filename (str): Nombre del archivo para guardar la imagen.
        """
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
        plt.colorbar(scatter)
        plt.title('K-means Clustering')
        plt.savefig(f"{self.base_path}/{filename}")
        plt.show()
        plt.close()

    def analyze(self, features: List[str], target: str):
        """
        Realiza el análisis de clustering utilizando las características y el objetivo especificados.

        Args:
            features (List[str]): Lista de características para el clustering.
            target (str): Variable objetivo (no se usa directamente en el clustering, pero puede ser útil para análisis posteriores).
        """
        print(f"Analyzing features: {features} with target: {target}")
        
        # Método del codo y K-means
        centroids, labels, optimal_k = self.elbow_and_kmeans(features)
        
        print("K-means clustering completed.")
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Centroids: {centroids}")
        print(f"Number of points in each cluster: {np.bincount(labels)}")

        # Validación
        y_true = self.data[target]
        y_pred = labels

        # Métricas adicionales
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
