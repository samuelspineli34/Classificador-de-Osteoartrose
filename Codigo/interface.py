import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import base64
import io
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from dash.dependencies import Input, Output, State
from io import BytesIO
import plotly.graph_objs as go

# Define o caminho para o modelo
model_path = 'model.h5'

threadsEscFraca = [2, 4, 6, 8, 10, 12]
temposEscFraca = [0.2040, 0.3144, 0.3738, 0.4093, 0.6169, 0.7767]

# Dados
threads_forte = [2, 4, 6, 8, 10, 12, 14]
tempos_forte = [1.6833, 1.0672, 0.8853, 0.7243, 0.7239, 0.7006, 0.7143]

# Carrega o modelo
model = load_model(model_path)

# Define os rótulos das classes
class_labels = ['Normal', 'Duvidoso', 'Moderado', 'Avançado', 'Severo']

app = dash.Dash(external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP])
server = app.server



imgUpload = dcc.Upload(
    id='upload-image',
    children=html.Div([
        html.A(
            html.I(className="bi bi-file-earmark-arrow-up", style={"fontSize": "50px", "align": "center", "margin": "20px", "padding": "20px"}),
            style={'width': '100%', 'height': '100%'},
        )
    ]),
    style={'width': '0', 'height': '0'},
    multiple=False
)

toastPrevisao = dbc.Toast(
    [html.P(id='prediction-output', className="mb-0")],
    header="Classificação",
)

######METRICAS PAI#######

row1 = html.Tr([html.Td("T.Execução"), html.Td("30.32m"), html.Td("32.91m")])
row2 = html.Tr([html.Td("Acurácia"), html.Td("0.4848"), html.Td("0.4848")])
row3 = html.Tr([html.Td("Precisão"), html.Td("0.4871"), html.Td("0.4994")])
row4 = html.Tr([html.Td("Recall"), html.Td("0.4848"), html.Td("0.4848")])
row5 = html.Tr([html.Td("F1-Score"), html.Td("0.4823"), html.Td("0.4562")])
row6 = html.Tr([html.Td("Especificidade"), html.Td("0.4848"), html.Td("0.4848")])

table_body = [html.Tbody([row1, row2, row3, row4, row5])]



tablePAI = [
    html.Thead(html.Tr([html.Th("Metricas"), html.Th("Normal"), html.Th("Paralelo")]))
]

tablePAIColuns = dbc.Table(tablePAI + table_body, bordered=True)



cardPAI = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(label="Metricas", tab_id="Tab-1PAI"),
                    dbc.Tab(label="Matriz de Confusão", tab_id="Tab-2PAI"),
                    dbc.Tab(label="Curva ROC", tab_id="Tab-3PAI"),
                   # dbc.Tab(label="Índice Jaccard", tab_id="Tab-4PAI"),
                ],
                id="cardPAI-tabs",
                active_tab="tab-1",
            )
        ),
        dbc.CardBody(html.P(id="card-contentPAI", className="card-textPAI")),
    ]
)

@app.callback(
    Output("card-contentPAI", "children"), [Input("cardPAI-tabs", "active_tab")]
)
def tab_contentPAI(active_tab):
    if active_tab == "Tab-1PAI":
        return tablePAIColuns
    elif active_tab == "Tab-2PAI":
        return html.Img(src='assets/matriz_confusao.png', style={'width': '100%'})
    elif active_tab == "Tab-3PAI":
        return html.Img(src='assets/curva_roc.png', style={'width': '100%'})

    else:
        return "Tab desconhecida"

#####METRICAS PARALELA########

row1 = html.Tr([html.Td("T.Treinamento"), html.Td("32.91m")])
row2 = html.Tr([html.Td("Speed-Up"), html.Td("0.9213")])


table_body = [html.Tbody([row1, row2])]


tablePARALELA = [
    html.Thead(html.Tr([html.Th("Metricas"), html.Th("Resultados")]))
]

tablePARALELAColuns = dbc.Table(tablePARALELA + table_body, bordered=True)

cardPARALELA = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(label="Metricas", tab_id="Tab-1PARALELA"),
                    dbc.Tab(label="Esc. Fraca", tab_id="Tab-2PARALELA"),
                    dbc.Tab(label="Esc. Forte", tab_id="Tab-3PARALELA"),
                ],
                id="cardPARALELA-tabs",
                active_tab="tab-1",
            )
        ),
        dbc.CardBody(html.P(id="card-contentPARALELA", className="card-textPARALELA")),
    ]
)

@app.callback(
    Output("card-contentPARALELA", "children"), [Input("cardPARALELA-tabs", "active_tab")]
)
def tab_contentPARALELA(active_tab):
    if active_tab == "Tab-1PARALELA":
        return tablePARALELAColuns
    elif active_tab == "Tab-2PARALELA":
        return html.Button("Atualizar", id="btn"), dcc.Graph(id='GraficoEscFraca')
    elif active_tab == "Tab-3PARALELA":
        return html.Button("Atualizar", id="btn"),dcc.Graph(id='graph')
    elif active_tab == "Tab-4PARALELA":
        return "Conteúdo da Tab 2PAI"
    else:
        return "Tab desconhecida"

######METRICAS DISTRIBUIDA########


cardDistribuida = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(label="Tempo de CPU", tab_id="tab-1Distribuida"),
                    dbc.Tab(label="Uso de Memoria", tab_id="tab-2Distribuida"),
                    dbc.Tab(label="Tempo de resposta", tab_id="tab-3Distribuida"),
                    dbc.Tab(label="Requests", tab_id="tab-4Distribuida"),

                ],
                id="cardDistribuida-tabs",
                active_tab="tab-1",
            )
        ),
        dbc.CardBody(html.P(id="card-contentDistribuida", className="card-textDistribuida")),
    ]
)

# Lista de conteúdos dos cartões
card_contents = [
    {
        "header": "Card 1",
        "title": "Original",
        "text": "Imagem original fornecida pelo usuário",
        "image": None,# Adicione a chave "image" com o valor None para o Card 1
        "id": "card-Card 1-image"
    },
    {
        "header": "Card 2",
        "title": "Redimensionada",
        "text": "Imagem redimensionada para 256x256",
    },
    {
        "header": "Card 3",
        "title": "Escala de cinza",
        "text": "Imagem convertida para escala de cinza"
    },
    {
        "header": "Card 4",
        "title": "Normalizada",
        "text": "Imagem normalizada para o modelo"
    },
    # Adicione mais dicionários para mais cartões
]

# Criação dos cartões
cards = []
for content in card_contents:
    card = dbc.Col(  # Coloque cada cartão em uma coluna
        dbc.Card(
            [
                dbc.CardHeader(content["header"]),
                dbc.CardBody(
                    [
                        html.H5(content["title"], className="card-title"),
                        html.P(content["text"], className="card-text"),
                        html.Img(id=f"card-{content['header']}-image", src=content.get("image", ""), style={'width': '100%'})
                    ]
                ),
            ],
            color="primary",
            inverse=True,
            style={"margin": "5px"}
        ),
        md=3  # Ajuste o tamanho da coluna conforme necessário
    )
    cards.append(card)

# Agora, 'cards' é uma lista de cartões que você pode adicionar ao seu layout

collapse = html.Div(
    [
        dbc.Button(
            html.I(className="bi bi-box-arrow-right"),
            id="horizontal-collapse-button",
            className="mb-3",
            color="primary",
            n_clicks=0,
        ),
        html.Div(
            dbc.Collapse(
                dbc.Card(
                    dbc.CardBody(
                        dbc.Row([
                            dbc.Col([
                                imgUpload
                            ])
                        ])
                    ),
                    style={"width": "300px","height": "100vh", "margin": "3px","z-index": "1000","background-color": "rgba(50, 56, 62, 0.7)"}
                ),
                id="horizontal-collapse",
                is_open=False,
                dimension="width",
            ),
            style={"minHeight": "100px","rgba": "50, 56, 62, 0.5"}
        ),
    ]
)


app.layout = html.Div([
        dbc.Row([
            dbc.Col([
                collapse,
            ], sm=1),
            dbc.Col([
                dbc.Row(cards),  # Coloque todas as colunas em uma linha

            ], sm=11)
        ]),
        dbc.Row([
            toastPrevisao
        ]),
        dbc.Row([
            dbc.Col([
                cardPAI
            ], sm=4),
            dbc.Col([
                cardPARALELA
            ], sm=4),
            dbc.Col([
                cardDistribuida
            ], sm=4)
        ]),

])



@app.callback(
    [Output('card-Card 1-image', 'src'),
     Output('card-Card 2-image', 'src'),
     Output('card-Card 3-image', 'src'),
     Output('card-Card 4-image', 'src'),
     Output('prediction-output', 'children')],
    [Input('upload-image', 'contents')])
def update_output(contents):
    if contents:
        # Obtém o tipo de conteúdo e o conteúdo codificado em base64
        content_type, content_string = contents.split(',')

        # Decodifica a string base64
        decoded = base64.b64decode(content_string)

        # Abre a imagem
        image = Image.open(io.BytesIO(decoded))

        # Redimensiona a imagem para o tamanho de entrada do modelo
        resized_image = image.resize((256, 256))

        # Converte a imagem para escala de cinza
        grayscale_image = resized_image.convert('L')

        # Converte a imagem em escala de cinza para um array numpy
        image_np = np.array(grayscale_image)

        # Normaliza os dados da imagem
        image_np = image_np.astype('float32') / 255.0

        # Adiciona uma nova dimensão à imagem para corresponder ao formato de entrada do modelo
        image_np = np.expand_dims(image_np, axis=-1)

        # Faz a previsão
        prediction = model.predict(np.expand_dims(image_np, axis=0))

        # Obtém a classe prevista
        predicted_class = class_labels[np.argmax(prediction)]

        # Codifica as imagens para base64 para exibição
        original_image_b64 = image_to_b64(image)
        resized_image_b64 = image_to_b64(resized_image)
        grayscale_image_b64 = image_to_b64(grayscale_image)

        # Escala a imagem normalizada de volta para 0-255 e converte para uint8 para exibição
        final_image = Image.fromarray((np.squeeze(image_np, axis=-1) * 255).astype('uint8'))
        final_image_b64 = image_to_b64(final_image)

        return (f'data:image/png;base64,{original_image_b64}',
                f'data:image/png;base64,{resized_image_b64}',
                f'data:image/png;base64,{grayscale_image_b64}',
                f'data:image/png;base64,{final_image_b64}',
                f'Classe prevista: {predicted_class}')

    return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update)
def image_to_b64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()



@app.callback(
    Output("card-contentPARALELA", "childrenPARALELA"), [Input("cardPARALELA-tabs", "active_tabPARALELA")]
)
def tab_content(active_tab):
    return "This is tab {}".format(active_tab)

@app.callback(
    Output("card-contentDistribuida", "childrenDistribuida"), [Input("cardDistribuida-tabs", "active_tabDistribuida")]
)
def tab_content(active_tab):
    return "This is tab {}".format(active_tab)

# Callback para atualizar o gráfico quando o botão for clicado
@app.callback(
    Output('GraficoEscFraca', 'figure'),
    [Input('btn', 'n_clicks')]
)
def update_graph(n_clicks):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=threadsEscFraca,
        y=temposEscFraca,
        mode='lines+markers',
        marker=dict(size=10),
        line=dict(width=2)
    ))
    fig.update_layout(
        title='Escalabilidade Fraca',
        xaxis=dict(title='Threads'),
        yaxis=dict(title='Tempo de Execução'),
        showlegend=False
    )
    return fig

# Callback para atualizar o gráfico quando o botão for clicado
@app.callback(
    Output('graph', 'figure'),
    [Input('btn', 'n_clicks')]
)
def update_graph(n_clicks):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=threads_forte,
        y=tempos_forte,
        mode='lines+markers',
        marker=dict(size=10),
        line=dict(width=2)
    ))
    fig.update_layout(
        title='Escalabilidade Forte',
        xaxis=dict(title='Threads'),
        yaxis=dict(title='Tempo de Execução'),
        showlegend=False
    )
    return fig


@app.callback(
    Output("horizontal-collapse", "is_open"),
    [Input("horizontal-collapse-button", "n_clicks")],
    [State("horizontal-collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == "__main__":
    app.run_server()