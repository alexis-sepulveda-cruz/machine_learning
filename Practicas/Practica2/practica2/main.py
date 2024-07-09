from practica2.regression.asset_regression import AssetRegression
from practica2.utils.util import get_data, filter_data_by_year
from practica2.utils.column_assets import ColumnAssets as Column


if __name__ == "__main__":
    # Supongamos que el DataFrame ya está cargado en la variable df
    df = get_data('stock_market_dataset.csv')

    # Crear una instancia de AssetRegression
    asset_reg = AssetRegression(df)

    # Filtrar datos para el año 2023
    df_2023 = filter_data_by_year(df, 2023)

    # Todas las columnas de activos
    all_assets_col = df_2023.columns.difference([Column.DATE, Column.YEAR, Column.MONTH, Column.YEAR_MONTH])

    # Realizar regresión lineal para un activo
    asset_reg.linear_regression(df_2023, Column.YEAR_MONTH, [Column.APPLE_PRICE, Column.AMAZON_PRICE])
    # asset_reg.linear_regression(df_2023, Column.YEAR_MONTH, all_assets_col)

    # Realizar regresión Ridge para un activo con diferentes alphas
    alphas = [0.1, 1, 10]
    asset_reg.ridge_regression(df_2023, Column.YEAR_MONTH, [Column.APPLE_PRICE, Column.AMAZON_PRICE], alphas)

    # Realizar regresión Lasso para un activo con diferentes alphas
    asset_reg.lasso_regression(df_2023, Column.YEAR_MONTH, [Column.APPLE_PRICE, Column.AMAZON_PRICE], alphas)

    # Realizar regresión polinomial para un activo con diferentes grados
    degrees = [2, 3, 4]
    asset_reg.polynomial_regression(df_2023, Column.YEAR_MONTH, [Column.APPLE_PRICE, Column.AMAZON_PRICE], degrees)

    # Realizar regresión polinomial con Ridge para un activo con diferentes grados y alphas
    asset_reg.polynomial_ridge_regression(df_2023, Column.YEAR_MONTH, [Column.APPLE_PRICE, Column.AMAZON_PRICE], degrees, alphas)

    # Realizar regresión polinomial con Lasso para un activo con diferentes grados y alphas
    asset_reg.polynomial_lasso_regression(df_2023, Column.YEAR_MONTH, [Column.APPLE_PRICE, Column.AMAZON_PRICE], degrees, alphas)

    # Realizar regresión KNN para un activo con diferentes valores de n_neighbors
    n_neighbors = [3, 5, 7]
    asset_reg.knn_regression(df_2023, Column.YEAR_MONTH, [Column.APPLE_PRICE, Column.AMAZON_PRICE], n_neighbors)

    # Realizar regresión con árbol de decisión para un activo con diferentes profundidades máximas
    max_depths = [3, 5, 7]
    asset_reg.decision_tree_regression(df_2023, Column.YEAR_MONTH, [Column.APPLE_PRICE, Column.AMAZON_PRICE], max_depths)