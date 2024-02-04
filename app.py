import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import dash_loading_spinners as dls
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
from preprocessing import preprocess_data
from visualization import generate_table

# Wczytanie danych
df = pd.read_csv('messy_data.csv', skipinitialspace=True)
df_cleaned = preprocess_data(df)

# Styl CSS dla elementów dashboardu
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Układ dashboardu
app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Img(src='https://dash.plotly.com/assets/images/language_icons/python_50px.svg', style={'height': '50px', 'verticalAlign': 'middle'}),
                        html.H1("Projekt PAD", style={'textAlign': 'center', 'color': 'white', 'marginBottom': '0', 'fontFamily': 'Roboto, sans-serif', 'fontWeight': '900', 'backgroundColor': 'black', 'display': 'inline-block', 'marginLeft': '10px'})
                    ],
                    style={'display': 'flex', 'alignItems': 'center', 'backgroundColor': 'black'}
                ),
                html.H2("Michał Czapliński (s32071)", style={'textAlign': 'left', 'color': 'white', 'fontFamily': 'Roboto, sans-serif', 'marginTop': '10px', 'fontSize': '25px'}),
                html.Hr(style={'width': '100%', 'border': '1px solid white', 'color': 'white', 'marginTop': '40px'}), # Zmiana grubości na 1px
                html.H2("1. Wstępna analiza danych", style={'textAlign': 'left', 'color': 'white', 'fontFamily': 'Roboto, sans-serif', 'marginTop': '10px', 'fontWeight': '700', 'fontSize': '20px'}),
            ]
        ),
        dcc.Loading(
            id="loading",
            children=[
                html.Div(id="data-table-container")
            ],
            type="default",
            fullscreen=True,
        ),
        html.Hr(style={'width': '100%', 'border': '1px solid white', 'color': 'white', 'marginTop': '40px'}), # Zmiana grubości na 1px
        html.H2("2. Wizualizacja rozkładu zmiennych, zależności ceny od innych zmiennych, liczebność kategorii", style={'textAlign': 'left', 'color': 'white', 'fontFamily': 'Roboto, sans-serif', 'marginTop': '10px', 'fontWeight': '700', 'fontSize': '20px'}),
        dcc.Dropdown(
            id='visualization-selector',
            options=[
                {'label': 'Histogramy zmiennych', 'value': 'histograms'},
                {'label': 'Zależność ceny od innych zmiennych', 'value': 'pairplot'},
                {'label': 'Liczebność kategorii', 'value': 'countplot'}
            ],
            value='histograms'
        ),
        html.Div(id='visualization-container'),
        html.Hr(style={'width': '100%', 'border': '1px solid white', 'color': 'white', 'marginTop': '40px'}), # Zmiana grubości na 1px
        html.H1("3. Budowa modelu regresji price od zmiennych carat, x dimension, depth", style={'textAlign': 'left', 'color': 'white', 'fontFamily': 'Roboto, sans-serif', 'marginTop': '10px', 'fontWeight': '700', 'fontSize': '20px'}),
        dcc.Dropdown(
            id='regression-variable-selector',
            options=[
                {'label': 'carat', 'value': 'carat'},
                {'label': 'x dimension', 'value': 'x dimension'},
                {'label': 'depth', 'value': 'depth'}
            ],
            value='carat'
        ),
        html.Div(id='regression-container')
    ],
    style={ 'margin': '0', 'backgroundColor': 'black', 'padding': '5%'}  
)

# Dodanie callbacków
@app.callback(
    Output("data-table-container", "children"),
    [Input("loading", "children")]
)
def display_table(children):
    return generate_table(df_cleaned)

@app.callback(
    Output('visualization-container', 'children'),
    [Input('visualization-selector', 'value')]
)
def update_visualization(selected_visualization):
    if selected_visualization == 'histograms':
        histograms = []
        for column in df_cleaned.columns:
            fig = px.histogram(df_cleaned, x=column, title=f'Rozkład zmiennej {column}')
            fig.update_layout(template='plotly_dark')  # Dodanie ciemnego tła
            histograms.append(dcc.Graph(figure=fig, id=f'histogram-{column}'))
        return [html.Div(histogram) for histogram in histograms]
    elif selected_visualization == 'pairplot':
        pairplot = px.scatter_matrix(df_cleaned, dimensions=['carat', 'clarity', 'color', 'cut', 'price'])
        pairplot.update_layout(title='Zależność ceny od innych zmiennych')
        pairplot.update_layout(template='plotly_dark')  # Dodanie ciemnego tła
        return dcc.Graph(figure=pairplot, id='pairplot')
    elif selected_visualization == 'countplot':
        countplot = px.bar(df_cleaned, x='cut', title='Liczebność kategorii')
        countplot.update_layout(template='plotly_dark')  # Dodanie ciemnego tła
        return dcc.Graph(figure=countplot, id='countplot')

@app.callback(
    Output('regression-container', 'children'),
    [Input('regression-variable-selector', 'value')]
)
def update_regression(selected_variable):
    # Budowa modelu regresji
    X = df_cleaned[[selected_variable]]
    y = df_cleaned['price']
    model = LinearRegression()
    model.fit(X, y)

    # Predykcja wartości
    y_pred = model.predict(X)

    # Wykres modelu regresji
    fig = px.scatter(x=X[selected_variable], y=y, title=f'Model Regresji: Cena vs {selected_variable}', labels={'x': selected_variable, 'y': 'Price'})
    fig.add_scatter(x=X[selected_variable], y=y_pred, mode='lines', name='Regression Line')
    fig.update_layout(template='plotly_dark')  # Dodanie ciemnego tła
    return dcc.Graph(figure=fig)

if __name__ == '__main__':
    app.run_server(debug=True, port=8888)
