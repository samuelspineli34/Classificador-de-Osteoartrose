import base64
import datetime
import io
import os
import numpy as np
from PIL import Image
import tensorflow as tf

import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output, State, clientside_callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# Load the model
model_path = './model.h5'
model = tf.keras.models.load_model(model_path)

# Define the class labels
class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])

color_mode_switch = html.Span(
    [
        dbc.Label(className="fa fa-moon", html_for="switch"),
        dbc.Switch( id="switch", value=True, className="d-inline-block ms-1", persistence=True),
        dbc.Label(className="fa fa-sun", html_for="switch"),
    ]
)

# Adicione um novo gráfico para exibir as métricas reais
app.layout = html.Div([
    html.H1('Classificador de níveis de Osteoartrose'),
    dbc.Row([
        color_mode_switch,
        dbc.Col(
            [
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        html.H4('Arraste a imagem ou selecione um arquivo (.jpg .png)')
                    ]),
                    style={
                        'width': '100%',
                        'height': '90%',
                        'lineHeight': '300%',
                        'borderWidth': '1px',
                        'borderStyle': 'solid',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px',
                        'textColor': 'black',
                    },
                    # Allow multiple files to be uploaded
                    multiple=False
                ),
                dcc.Slider(
                    id='my-slider',
                    min=1,
                    max=20,
                    step=1,
                    value=10,
                    marks={i: str(i) for i in range(21)},
                ),
                html.Div(id='slider-output-container'),
                dcc.Graph(id='metrics-graph'),  # Novo gráfico para métricas reais
                dcc.Dropdown(
                    id='demo-dropdown',
                    options=[
                        {'label': 'Gráfico de Barras', 'value': 'bar'},
                        {'label': 'Gráfico de Pizza', 'value': 'pie'}
                    ],
                    value='bar',
                    style={'width': '100%',
                           'lineHeight': '100%',
                           'textAlign': 'center',
                           'marginTop': '5px',
                           'textColor': 'black',
                           'backgroundColor': 'white'
                           }
                ),
                html.Div(id='dd-output-container', ),

            ],
            width=4,style={'backgroundColor':'light_blue',
                           'margin': '10px',}
        ),
        dbc.Col(
            [
                html.Div(id='output-data-upload',
                         style={
                             'width': '150%',
                             'height': '100%',
                             'lineHeight': '100%',
                             'borderWidth': '1px',
                             'borderStyle': 'solid',
                             'borderRadius': '5px',
                             'textAlign': 'center',
                         }),
            ],
            width=3
        ),
    ]),
])

# Mantenha um histórico das métricas
metric_history = []

# Mantenha um contador de True Positives
true_positives_count = 0

# Atualize o contador de True Positives quando uma nova imagem for processada
def update_true_positives_count(image_bytes):
    global true_positives_count
    predicted_class = classify_image(image_bytes)
    if predicted_class == "A imagem possui osteoartrose.":
        true_positives_count += 1

# Atualize o callback update_metrics_graph
@app.callback(
    Output('metrics-graph', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    State('metrics-graph', 'figure'))
def update_metrics_graph(contents, filename, date, existing_figure):
    global true_positives_count

    # Atualize o contador de True Positives se uma nova imagem for processada
    if contents is not None:
        update_true_positives_count(contents.encode('utf-8'))

    # Adicione as métricas ao histórico
    false_positives = 0  # Defina false_positives e outros valores conforme necessário
    false_negatives = 0
    true_negatives = 0
    f_score = 0

    # Adicione as métricas ao histórico
    metric_history.append({'True Positives': true_positives_count, 'False Positives': false_positives,
                           'False Negatives': false_negatives, 'True Negatives': true_negatives,
                           'F-score': f_score})

    # Limita o número de pontos no gráfico para 20
    if len(metric_history) > 20:
        metric_history.pop(0)  # Remove o primeiro ponto do histórico

    # Atualize os dados do gráfico com base no histórico
    x_values = list(metric_history[0].keys())  # Usa as chaves do primeiro dicionário como x
    data = []
    for metric_name in x_values:
        y_values = [metric[metric_name] for metric in metric_history]
        data.append(go.Bar(x=list(range(1, len(metric_history) + 1)), y=y_values,
                           name=metric_name))

    # Retorna a figura atualizada
    updated_figure = {
        'data': data,
        'layout': go.Layout(
            title='Métricas de Avaliação',
            yaxis=dict(title='Valor'),
        )
    }

    return updated_figure

#Function to classify the uploaded image
def classify_image(image_bytes):
    # Load and preprocess the image
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((256, 256))
    image = image.convert('L')
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    # Make the prediction
    prediction = model.predict(image)

    # Get the predicted class label
    predicted_class = class_labels[np.argmax(prediction)]

    # Return message indicating whether the image has osteoarthritis or not
    if predicted_class == 'Class 0':  # Adjust this condition based on your model's output
        return "A imagem não possui osteoartrose."
    else:
        return "A imagem possui osteoartrose."

#Parse de dados
def parse_contents(contents, filename, date):
    if contents is None:
        return html.Div(['Sem arquivo selecionado'])

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        if 'image' in content_type:
            # Construindo o src da imagem com base no conteúdo
            src = 'data:image/png;base64,' + base64.b64encode(decoded).decode()

            # Make prediction on the uploaded image
            osteoarthritis_message = classify_image(decoded)

            # Retorna a div com a imagem e a mensagem sobre a osteoartrose
            return html.Div([
                html.H5(filename),
                html.Img(src=src),
                html.H6(datetime.datetime.fromtimestamp(date)),
                html.P(osteoarthritis_message)
            ])
        elif 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            # Restante do código...
    except Exception as e:
        print(e)
        return html.Div(['Houve um erro ao processar o arquivo.'])

#Callback do upload da imagem (alterar valores da propriedade da imagem para valores pertinentes ao trabalho)
@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(contents, filename, date):
    # Adicionando instruções de impressão para depurar
    print("Contents:", contents)
    print("Filename:", filename)
    print("Date:", date)

    if contents is not None:
        children = parse_contents(contents, filename, date)
        return children
    else:
        return html.Div(['Sem arquivo selecionado'])


#Callback do slide (usar pra testes)
@app.callback(
    Output('slider-output-container', 'children'),
    Input('my-slider', 'value'))
def update_output(value):
    return "" #'Número de pontos no gráfico "{}"'.format(value)


#Callback do gráfico
@app.callback(
    Output('graph1', 'figure'),
    [Input('my-slider', 'value')])
def update_graph1(value):
    return {
        'data': [go.Bar(x=['A', 'B', 'C'], y=[value, value+1, value+2])],
        'layout': go.Layout(title='Gráfico 1')
    }

@app.callback(
    Output('graph2', 'figure'),
    [Input('my-slider', 'value')])
def update_graph2(value):
    return {
        'data': [go.Scatter(x=[1, 2, 3], y=[value, value*2, value*3])],
        'layout': go.Layout(title='Gráfico 2')
    }

@app.callback(
    Output('graph3', 'figure'),
    [Input('my-slider', 'value')])
def update_graph3(value):
    return {
        'data': [go.Pie(labels=['Label 1', 'Label 2', 'Label 3'], values=[value, value+1, value+2])],
        'layout': go.Layout(title='Gráfico 3')
    }
#Callback do botão
@app.callback(
    Output('dd-output-container', 'children'),
    Input('demo-dropdown', 'value')
)
def update_output(value):
    return #f'You have selected {value}'

@app.callback(
    Output('pie-chart', 'figure'),
    Input('my-slider', 'value'))
def update_pie_chart(value):
    labels = ['Usage', 'Free']
    values = [value, 100-value]  # Valores falsos

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    return fig

@app.callback(
    Output('classification-output', 'children'),
    [Input('upload-data', 'contents')])
def update_classification(contents):
    if contents is not None:
        # Faça sua lógica de classificação aqui
        classification = 'Classificação Falsa'  # Valor falso
        return classification
    else:
        return 'Sem arquivo selecionado'

clientside_callback(
    """
    (switchOn) => {
       document.documentElement.setAttribute("data-bs-theme", switchOn ? "light" : "dark");
       return window.dash_clientside.no_update
    }
    """,
    Output("switch", "id"),
    Input("switch", "value"),
)

if __name__ == '__main__':
    app.run_server(debug=True)

