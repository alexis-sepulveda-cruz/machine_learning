from pia.util_csv_reader import UtilCSVReader
from pia.clasification import Classification

if __name__ == "__main__":
    # # Crear una instancia de la clase
    # csv_reader = UtilCSVReader()

    # # Leer el archivo CSV y obtener el DataFrame
    # df_train = csv_reader.read_csv_to_dataframe('train.csv')
    # df_test = csv_reader.read_csv_to_dataframe('test.csv')
    # print("--------- Datos train ---------")
    # print(df_train.head())
    # print("--------- Datos test ---------")
    # print(df_test.head())
    classification: Classification = Classification()
    classification.run_classification()