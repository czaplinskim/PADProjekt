import pandas as pd
import numpy as np

def preprocess_data(dataframe):
    # Usunięcie duplikatów
    dataframe = dataframe.drop_duplicates()

    # Wybór tylko kolumn numerycznych
    numeric_cols = dataframe.select_dtypes(include=np.number).columns
    
    # Uzupełnienie brakujących wartości w kolumnach numerycznych za pomocą mediany
    dataframe[numeric_cols] = dataframe[numeric_cols].fillna(dataframe[numeric_cols].median())

    dataframe.dropna(subset=['price', 'x dimension'], inplace=True)

    # Usunięcie wierszy, w których brakuje jakichkolwiek danych
    dataframe = dataframe.dropna()
    
    # Usunięcie wartości odstających
    
    # Sprawdzenie spójności danych
    
    # Sprawdzenie braków w danych
    missing_values = dataframe.isnull().sum()
    print("Liczba brakujących wartości w kolumnach po oczyszczeniu:")
    print(missing_values)
    
    # Skala wartości - można przeprowadzić normalizację lub standaryzację
    
    return dataframe
