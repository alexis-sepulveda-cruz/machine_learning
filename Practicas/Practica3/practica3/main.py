from practica3.classification.asset_classification import AssetClassification
from practica3.utils.util import get_data, print_separator
from practica3.utils.column_assets import ColumnAssets as Column


if __name__ == "__main__":
    # Supongamos que el DataFrame ya está cargado en la variable df
    df = get_data('stock_market_dataset.csv')

    # Genera una instancia de análisis de activos por clasificación
    classifier = AssetClassification(df)

    # Definir las características y la variable objetivo
    features = [Column.APPLE_PRICE, Column.AMAZON_PRICE]
    target = Column.YEAR

    # Ejecutar KNN
    classifier.knn_classifier(features, target, n_neighbors=3)
    print_separator()

    # Ejecutar Regresión Logística
    classifier.logistic_regression(features, target)
    print_separator()

    # Ejecutar SVM
    classifier.svm_classifier(features, target)
    print_separator()

    # Ejecutar Árbol de Decisión
    classifier.decision_tree(features, target)
