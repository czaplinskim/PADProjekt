import plotly.express as px
from dash import html

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Nagłówki kolumn
        [html.Tr([html.Th(col, style={'color': 'white'}) for col in dataframe.columns])] +  # Zmiana koloru tekstu na biały
        # Wiersze z danymi
        [html.Tr([html.Td(dataframe.iloc[i][col], style={'color': 'white'}) for col in dataframe.columns]) for i in range(min(len(dataframe), max_rows))]  # Zmiana koloru tekstu na biały
    )
