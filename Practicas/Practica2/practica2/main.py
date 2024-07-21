from practica2.regression.asset_regression import AssetRegression
from practica2.utils.util import get_data, filter_data_by_year, print_separator
from practica2.utils.column_assets import ColumnAssets as Column

    
if __name__ == "__main__":
    # Supongamos que el DataFrame ya está cargado en la variable df
    df = get_data('stock_market_dataset.csv')

    # Filtrar datos para el año 2023
    df = filter_data_by_year(df, 2023)

    # Crear una instancia de AssetRegression
    asset_reg = AssetRegression(df)

    # # Seleccionar columnas selectas
    # targets = [Column.APPLE_PRICE, Column.AMAZON_PRICE]

    # # Todas las columnas de activos
    # all_assets_col = df.columns.difference([Column.DATE, Column.YEAR, Column.MONTH, Column.YEAR_MONTH])

    # # Realizar regresión lineal para un activo
    # asset_reg.linear_regression(df, Column.YEAR_MONTH, targets)
    # # asset_reg.linear_regression(df_2023, Column.YEAR_MONTH, all_assets_col)
    # print_separator()

    # # Realizar regresión Ridge para un activo con diferentes alphas
    # alphas = [0.1, 1, 10]
    # asset_reg.ridge_regression(df, Column.YEAR_MONTH, targets, alphas)
    # print_separator()

    # # Realizar regresión Lasso para un activo con diferentes alphas
    # asset_reg.lasso_regression(df, Column.YEAR_MONTH, targets, alphas)
    # print_separator()

    # # Realizar regresión polinomial para un activo con diferentes grados
    # degrees = [2, 3, 4]
    # asset_reg.polynomial_regression(df, Column.YEAR_MONTH, targets, degrees)
    # print_separator()

    # # Realizar regresión polinomial con Ridge para un activo con diferentes grados y alphas
    # asset_reg.polynomial_ridge_regression(df, Column.YEAR_MONTH, targets, degrees, alphas)
    # print_separator()

    # # Realizar regresión polinomial con Lasso para un activo con diferentes grados y alphas
    # asset_reg.polynomial_lasso_regression(df, Column.YEAR_MONTH, targets, degrees, alphas)
    # print_separator()

    # # Realizar regresión KNN para un activo con diferentes valores de n_neighbors
    # n_neighbors = [3, 5, 7]
    # asset_reg.knn_regression(df, Column.YEAR_MONTH, targets, n_neighbors)
    # print_separator()

    # # Realizar regresión con árbol de decisión para un activo con diferentes profundidades máximas
    # max_depths = [3, 5, 7]
    # asset_reg.decision_tree_regression(df, Column.YEAR_MONTH, targets, max_depths)
    # print_separator()

    # # Realizar y evaluar la regresión lineal para un activo
    # asset_reg.train_and_evaluate_model(df, Column.YEAR_MONTH, Column.APPLE_PRICE)
    # print_separator()

    # # Ejemplos de uso de las nuevas funciones
    # asset_reg.train_and_evaluate_ridge(df, Column.YEAR_MONTH, Column.APPLE_PRICE, alpha=1.0)
    # print_separator()
    # asset_reg.train_and_evaluate_lasso(df, Column.YEAR_MONTH, Column.APPLE_PRICE, alpha=1.0)
    # print_separator()
    # asset_reg.train_and_evaluate_polynomial(df, Column.YEAR_MONTH, Column.APPLE_PRICE, degree=2)
    # print_separator()
    # asset_reg.train_and_evaluate_polynomial_ridge(df, Column.YEAR_MONTH, Column.APPLE_PRICE, degree=2, alpha=1.0)
    # print_separator()
    # asset_reg.train_and_evaluate_polynomial_lasso(df, Column.YEAR_MONTH, Column.APPLE_PRICE, degree=2, alpha=1.0)
    # print_separator()
    # asset_reg.train_and_evaluate_knn(df, Column.YEAR_MONTH, Column.APPLE_PRICE, n_neighbors=5)
    # print_separator()
    # asset_reg.train_and_evaluate_decision_tree(df, Column.YEAR_MONTH, Column.APPLE_PRICE, max_depth=5)
    # print_separator()

    # Regresión lineal con búsqueda de cuadrícula
    # asset_reg.train_and_evaluate_linear(df, Column.YEAR_MONTH, Column.APPLE_PRICE, perform_grid_search=True)
    # print_separator()

    # # Regresión lineal Ridge con búsqueda de cuadrícula
    # asset_reg.train_and_evaluate_ridge(df, Column.YEAR_MONTH, Column.APPLE_PRICE, perform_grid_search=True)
    # print_separator()

    # # Regresión lineal Lasso con búsqueda de cuadrícula
    # asset_reg.train_and_evaluate_lasso(df, Column.YEAR_MONTH, Column.APPLE_PRICE, perform_grid_search=True)
    # print_separator()

    # Regresión polinomial con búsqueda de cuadrícula
    asset_reg.train_and_evaluate_polynomial(df, Column.YEAR_MONTH, Column.APPLE_PRICE, perform_grid_search=True)
    print_separator()

    # Regresión polinomial Ridge con búsqueda de cuadrícula
    asset_reg.train_and_evaluate_polynomial_ridge(df, Column.YEAR_MONTH, Column.APPLE_PRICE, perform_grid_search=True)
    print_separator()

    # Regresión polinomial Lasso con búsqueda de cuadrícula
    asset_reg.train_and_evaluate_polynomial_lasso(df, Column.YEAR_MONTH, Column.APPLE_PRICE, perform_grid_search=True)
    print_separator()

    # Regresión KNN con búsqueda de cuadrícula
    asset_reg.train_and_evaluate_knn(df, Column.YEAR_MONTH, Column.APPLE_PRICE, perform_grid_search=True)
    print_separator()

    # Regresión con árbol de decisión con búsqueda de cuadrícula
    asset_reg.train_and_evaluate_decision_tree(df, Column.YEAR_MONTH, Column.APPLE_PRICE, perform_grid_search=True)
    print_separator()


    