from practica4.cluster.asset_cluster import AssetCluster
from practica4.utils.util import get_data, print_separator
from practica4.utils.column_assets import ColumnAssets as Column


if __name__ == "__main__":
    # Supongamos que el DataFrame ya está cargado en la variable df
    df = get_data('stock_market_dataset.csv')

    # Genera una instancia de análisis de activos por cluster
    cluster = AssetCluster(df)

    # Definir las características y la variable objetivo
    features = [Column.APPLE_PRICE, Column.AMAZON_PRICE]
    target = Column.YEAR

    # Realizar el análisis
    cluster.analyze(features, target)