from dash import Dash,dash, dcc, html, dash_table
import pandas as pd
from dash.dependencies import Input, Output, State
import base64
import plotly.express as px
import io
from layout import survival_analysis_page, covariate_analysis_page, kaplan_meier_page, cox_regression_page, log_rank_page, display_logrank_summary_table, ver_dataset_page
from kaplan_meier import plot_kaplan_meier, plot_km_G, plot_km_disc
from cox_regression import run_cox_regression
from log_rank_test import perform_log_rank_test
import matplotlib.pyplot as plt
from ollama import Client
from ollama_AI import generate_explanation 
import plotly.graph_objs as go
from dash import callback_context

# Inicializar la aplicación Dash
app = Dash(__name__, suppress_callback_exceptions=True)

client = Client(
  host='http://127.0.0.1:11434',
  headers={'x-some-header': 'some-value'}
)

df = pd.read_csv(r"C:\Users\lucia\Desktop\5º\TFG\prepared OULAD dataset\data_final.csv", sep=';')
df_limpio = pd.read_csv(r'C:\Users\lucia\Desktop\5º\TFG\prepared OULAD dataset\dataset_limpio.csv', sep=';')


# Barra de navegación fija en la parte superior
navbar = html.Div([ 
    html.Div([ 
        html.A('INICIO', id='inicio-link', href='#', className='navbar-link'),  # Link en lugar de botón
        dcc.Link('VER DATASET', href='/ver-dataset', className='navbar-link'), 
        dcc.Link('ANÁLISIS DE COVARIABLES', href='/covariate-analysis', className='navbar-link'),
        dcc.Link('ANÁLISIS DE SUPERVIVENCIA', href='/survival-analysis', className='navbar-link'),
    ], className='navbar-links')
], id='navbar', style={'position': 'fixed', 'top': '0', 'left': '0', 'width': '100%', 'background-color': '#f4f7f6', 'padding': '10px', 'z-index': '1000'})

app.layout = html.Div([ 
    navbar, 
    dcc.Location(id='url', refresh=False),  
    html.Div(id='page-content'),  
    # Cuadro de confirmación
    dcc.ConfirmDialog(
        id='confirm-dialog',
        message='¿Estás seguro de que deseas volver a la página inicial? Perderás el dataset cargado y todo el análisis realizado.',
        displayed=False,  # Inicialmente no se muestra
        submit_n_clicks=0,  # Mantener el contador de clicks
        cancel_n_clicks=0  
    )
])
@app.callback(
    Output('confirm-dialog', 'displayed'),
    Input('inicio-link', 'n_clicks'),
    prevent_initial_call=True  # Evita que se active al cargar la página
)
def mostrar_confirmacion(n_clicks):
    if n_clicks:
        return True 
    return False

@app.callback(
    Output('url', 'pathname'),
    [Input('confirm-dialog', 'submit_n_clicks'),
    Input('confirm-dialog', 'cancel_n_clicks')],
    prevent_initial_call=True
)

def navegar_a_inicio(submit_n_clicks, cancel_n_clicks):

    ctx = callback_context  
    if not ctx.triggered:
        return dash.no_update  
    
    trigger_id = ctx.triggered[0]['prop_id']  
    #Aceptar
    if 'submit_n_clicks' in trigger_id:
        print("Aceptar clickeado")
        return '/'  # Redirige a la página de inicio
    
    # Cancelar"
    if 'cancel_n_clicks' in trigger_id:
        print("Cancelar clickeado")
        return dash.no_update  # No hacer nada
    
    return dash.no_update 

# Página de HOME
home_page = html.Div([ 
html.Div([ 
    html.Video(
        src='/assets/banner.mp4', 
        id='banner-video', 
        autoPlay=True, 
        muted=True, 
        loop=True, 
        style={
            'width': '100%', 
            'maxHeight': '350px', 
            'display': 'block', 
            'marginTop': '0px', 
            'marginBottom': '0px',
            'objectFit': 'cover'  
        }
    )
], id="banner-container", style={'width': '100%', 'padding': '0', 'margin': '0'}),
    
    html.Div([ 
        html.Img(src='/assets/datos.png', style={'height': '150px', 'marginTop': '10px', 'marginLeft': '20px', 'display': 'none'}), 
        html.H1("DASHBOARD 'SURVIVAL ANALYSIS'", style={'textAlign': 'center', 'fontSize': '2.5em', 'display': 'inline-block'})
    ], style={'textAlign': 'center', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '0px'}), 
    
    dcc.Loading(
        id="loading-spinner",
        type="circle",  
        children=html.Div([  
            html.H3("Cargar Dataset para preprocesarlo y analizarlo", id='upload-text'), 
            dcc.Upload(
                id='upload-data',
                children=html.Button('Sube tu CSV'),
                multiple=False
            ),
            html.Div(id='output-data-upload') 
        ], style={'textAlign': 'center', 'marginTop': '30px'}),
    ),
    
    # Botón para limpiar y cargar el dataset limpio
    html.Div([ 
        html.Button('Preprocesa CSV', id='load-clean', n_clicks=0),
    ], style={'textAlign': 'center', 'marginTop': '20px'}),
])
#ocultar frase incial: "Cargar Dataset..."
@app.callback(
    Output('upload-text', 'style'),  # Cambiar el estilo del texto
    [Input('upload-data', 'contents')]
)
def hide_upload_text(contents):
    # Si se ha cargado un archivo, ocultamos el texto
    if contents is not None:
        return {'display': 'none'}  # Ocultar el texto
    return {'display': 'block'} 

# Función para procesar el archivo cargado y mostrarlo en una tabla
def display_data(df, title):
    return html.Div([
        html.H5(title),
        dash_table.DataTable(
            id='data-table',
            columns=[{"name": col, "id": col} for col in df.columns],  
            data=df.to_dict('records'),  
            style_table={'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '15px'}, 
        ),
    ])
# Función para cargar el archivo CSV
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=";")
    return df.head(10000)  # Limitamos a las primeras 10000 filas por eficiencia

def verificar_archivo_correcto(contents, filename):
    # Compara el nombre del archivo cargado con el archivo esperado
    archivo_esperado = "temp_data.csv"
    
    # Verificar si el nombre del archivo cargado es el esperado
    if filename != archivo_esperado:
        return False
    return True


# Función para actualizar la página y mostrar el archivo cargado
@app.callback(
    [Output('upload-data', 'style'),
     Output('load-clean', 'style'), 
     Output('output-data-upload', 'children')], 
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'), 
     Input('load-clean', 'n_clicks')]
)

def update_output(contents,filename, n_clicks):
    if contents is None:
        return {'display': 'block'},{'display': 'none'}, html.Div(["No se ha cargado ningún archivo aún."],style={'marginTop': '20px', 'marginBottom': '0px'})
    
    if not verificar_archivo_correcto(contents, filename):
        return {'display': 'none'}, {'display': 'none'}, html.Div(
            [
                "ERROR: El archivo cargado no es el adecuado.",
                html.Br(), 
                "Este sistema es compatible únicamente con el dataset temp_data.csv"
            ],
            style={
                'color': 'red',       
                'fontSize': '20px',   
                'textAlign': 'center',
                'marginTop': '20px'   
            }
        )
    # Cargar el archivo CSV
    df = parse_contents(contents)
    
    if n_clicks > 0:
        #df_limpio = df.preprocess_data
        return {'display': 'none'},{'display': 'none'}, display_data(df_limpio, "Archivo Preprocesado")
    
    # Si no se ha presionado el botón de limpiar, mostrar el archivo bruto
    return {'display': 'none'},{'display': 'inline-block'}, display_data(df, "Archivo Bruto")


#maneja que no aparezca la barra de navegacion hasta que se limpie el dataset
@app.callback(
    Output('navbar', 'style'),
    [Input('load-clean', 'n_clicks')]
)
def toggle_navbar(n_clicks):
    if n_clicks > 0:
        return {'position': 'fixed', 'top': '0', 'left': '0', 'width': '100%', 'background-color': '#f4f7f6', 'padding': '10px', 'z-index': '1000'}
    return {'display': 'none'}  # Si no se ha presionado el botón, la barra de navegación permanece oculta

# Callbacks para manejar la navegación entre páginas
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/':
        return home_page
    elif pathname == '/covariate-analysis':
        return covariate_analysis_page
    elif pathname == '/survival-analysis':
        return survival_analysis_page  
    elif pathname == '/survival-analysis/kaplan-meier':
        return kaplan_meier_page 
    elif pathname == '/survival-analysis/cox-regression':
        return cox_regression_page
    elif pathname == '/survival-analysis/log-rank':
        return log_rank_page
    elif pathname == '/ver-dataset':  
        return ver_dataset_page
    else:
        return home_page

#maneja navegacion de kaplan
@app.callback(
    Output('km-cov-div', 'children'),
    [Input('botonG', 'n_clicks'), Input('botonDisc', 'n_clicks'), Input('botonNone', 'n_clicks')]
)
def update_km_cov(gender_clicks, disability_clicks, none_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return None  
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'botonG':
        return plot_km_G(df_limpio)  
    elif button_id == 'botonDisc':
        return plot_km_disc(df_limpio, group_by='disability_N') 
    elif button_id == 'botonNone':
        return None 

df_limpio['gender'] = df_limpio['gender_F'].map({1: 'Femenino', 0: 'Masculino'})
df_limpio['disability'] = df_limpio['disability_N'].map({1: 'Con Discapacidad', 0: 'Sin Discapacidad'})
df_limpio['abandono'] = df_limpio['final_result'].map({1: 'Abandono', 0: 'No abandono'})

@app.callback(
    Output('openai-answer-kaplan', 'value'),
    [Input('explicar-btn-kaplan', 'n_clicks')],
    [State('km-global-div', 'children')] 
)
def explicar_kaplan(n_clicks, kaplan_img):

    if n_clicks is not None and n_clicks > 0:
        # Verificar que la imagen de Kaplan-Meier ha sido generada antes de continuar
        if kaplan_img:
            prompt = (
                f"Dame una conclusión en español de las gráficas obtenidas de Kaplan-Meier: {kaplan_img}."
            )
            # Llamar a la IA para obtener la explicación
            respuesta = responder_pregunta_con_llama3(prompt)
            return respuesta
    return ""  # Si no se hace clic o no hay gráfica, retornar vacío

# Callback para actualizar el gráfico según la selección del Dropdown
@app.callback(
    [Output('covariables-graph', 'figure'),
    Output('graph-explanation', 'children')],
    [Input('covariables-dropdown', 'value')]
)
def update_graph(col_chosen):
    if col_chosen == 'abandono':
        conteo_abandono = df_limpio['final_result'].value_counts().sort_index()

        # Crear el gráfico con barras
        fig = go.Figure()

        # Contar las cantidades para cada categoría (No Abandono y Abandono)
        count_no_abandono = conteo_abandono[0] if 0 in conteo_abandono else 0
        count_abandono = conteo_abandono[1] if 1 in conteo_abandono else 0

        # Añadir las barras para No Abandono y Abandono
        fig.add_trace(go.Bar(
            x=['No Abandono'],  
            y=[count_no_abandono],  
            name='',
            marker_color='#1abc9c',
            hovertemplate='No Abandono: %{y}'
        ))

        fig.add_trace(go.Bar(
            x=['Abandono'],  
            y=[count_abandono],  
            name='',
            marker_color='#006400',
            hovertemplate='Abandono: %{y}'
        ))

        # Personalizar la apariencia del gráfico
        fig.update_layout(
            title='Abandono vs No Abandono',
            xaxis_title='Resultado Final',
            yaxis_title='Número de Estudiantes',
            barmode='group', 
            xaxis=dict(
                tickmode='array', 
                tickvals=['No Abandono', 'Abandono'],  
                ticktext=[f'No Abandono: {count_no_abandono}', f'Abandono: {count_abandono}'], 
            ),
            legend_title="Evento",
        )
    
        explicacion = (
            """
            Este gráfico muestra la distribución de los estudiantes en función de si han abandonado o no el curso. 
            La columna verde, representa a los estudiantes que no han abandonado, mientras que la
             morada a los que sí.
            Como se puede observar, la mayoría de los estudiantes no han llevado a cabo el evento de abandono 
            lo que sugiere que la tasa de abandono es relativamente baja en este cojunto de datos.
            """
        )     
           
        return fig, explicacion
    
    elif col_chosen == 'gender':
        conteo_genero = df_limpio['gender'].value_counts().sort_index()
        
        # Contamos cuántos estudiantes hay según su género y si han abandonado o no
        fig = px.histogram(df_limpio, x='gender', color='final_result', barmode='group', title='Abandono según Género',
                           color_discrete_map={0: '#1abc9c', 1: '#006400'})

        ticktext = [f'Masculino: {conteo_genero[0]}', f'Femenino: {conteo_genero[1]}']

        fig.update_layout(
            xaxis_title='Género', 
            yaxis_title='Número de Estudiantes', 
            legend_title="Abandono",
            xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=ticktext)  
        )
        
        explicacion = (
            """
            Este gráfico muestra la distribución del abandono influido por el género. Podemos concluir en que la mayoría de los estudiantes
            no han abandonado el curso y se observa una diferencia destacable en el abandono por género. Las estudiantes femeninas
            presentan una mayor proporción de abandono en comparación con los estudiantes masculinos, por ello, en este conjunto las mujeres tienen una
            tasa de abandono superior. A pesar de ello, en ambos sexos, el número de estudiantes 
            que no abandonan el curso es significativamente mayor que el número de los que si abandonan.
            
            """
        )
        return fig, explicacion
    
    elif col_chosen == 'disability':
        conteo_discapacidad = df_limpio['disability'].value_counts().sort_index()

        # Contamos cuántos estudiantes hay según si tienen discapacidad y si han abandonado o no
        fig = px.histogram(df_limpio, x='disability', color='final_result', barmode='group', title='Abandono según Discapacidad',
                           color_discrete_map={0: '#1abc9c', 1: '#006400'})

        ticktext = [f'Sin Discapacidad: {conteo_discapacidad[0]}', f'Con Discapacidad: {conteo_discapacidad[1]}']

        fig.update_layout(
        xaxis_title='Discapacidad', 
        yaxis_title='Número de Estudiantes', 
        legend_title="Abandono",
        xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=ticktext) 
    )

        explicacion = (
            """
            Este gráfico muestra la distribución del abandono influido por la discapacidad. A pesar de que el número de estudiantes sin discapacidad
            es considerablemente menor, la tasa de abandono en este conjunto es más alta en comparación con los estudiantes que presentan discapacidad.
            Aún así, la mayoría de estudiantes, tanto con como sin, no abandonan el curso, y es la conclusión que podemos destacar del grafo presente. Esto
            nos lleva a entender que aunque la discapacidad puede estar asociada a un riesgo superior de que suceda este evento, la diferencia no es 
            considerable para este conjunto de datos.
            
            """
        )
        return fig, explicacion

@app.callback(
    Output('cox-regression-output', 'children'),
    [Input('covariables-dropdown-cox', 'value')]
)
def update_cox_model(covariables):
    if covariables is None or len(covariables) == 0:
        return html.Div(["Selecciona al menos una covariable."])

    # Asegurarnos de que covariables sea una lista
    if isinstance(covariables, str):  
        covariables = [covariables]
        
    # Llamamos a la función de regresión de Cox con las covariables seleccionadas
    summary, cox_table_html = run_cox_regression(df_limpio, covariables)
    
    return cox_table_html

@app.callback(
    Output('openai-answer-cox', 'value'),
    [Input('btn-cox', 'n_clicks')],
    [State('cox-regression-output', 'children')] 
)
def explicar_cox(n_clicks, cox_content):
    if n_clicks is not None and n_clicks > 0:
        # Verificar que la tabla de Cox ha sido generada antes de continuar
        if cox_content:
            prompt = (
                f"Explica los resultados dela Regresion de Cox generados con los siguientes resultados:\n"
                f"Resultados: {cox_content}\n"
                f"¿Cómo afectan las covariables a la probabilidad de abandono?"
            )
            # Llamar a la IA para obtener la explicación
            respuesta = responder_pregunta_con_llama3(prompt)
            return respuesta
    return "" 

#callback para el control del analisis de log rank
@app.callback(
    Output('logrank-test-output', 'children'),
    [Input('covariables-dropdown-logrank', 'value')]
)

def update_logrank_test(covariables):
    # Verificar que al menos se haya seleccionado una covariable
    if not covariables:
        return html.Div(["Selecciona al menos una covariable para comparar."])

    panels = []

    # Ejecutar el Test de Log-Rank para cada covariable seleccionada
    for covariable in covariables:
        res_df = perform_log_rank_test(df_limpio, covariable)
        table = display_logrank_summary_table(res_df)
        panels.append(html.Div([html.H3(f"Resultado del Test de Log-Rank para {covariable}",
                                        style={'textAlign': 'center'}), table]))
        
    # Mostrar todos los resultados para las covariables seleccionadas
    return html.Div(panels)

@app.callback(
    Output('openai-answer-logrank', 'value'),
    [Input('explicar-btn-logrank', 'n_clicks')],
    [State('logrank-test-output', 'children')]  
)
def explicar_logrank(n_clicks, logrank_content):
    if n_clicks is not None and n_clicks > 0:
        # Verificar que la tabla del Test de Log-Rank ha sido generada antes de continuar
        if logrank_content:
            prompt = (
                #RESPUESTA LARGA
                #f"Explica los resultados del Test de Log-Rank generados con los siguientes resultados:\n"
                #RESPUESTA CORTA
                f"Me puedes dar una conclusión breve según los resultados obtenidos de Log Rank:\n"
                f"Resultados: {logrank_content}\n"
                #f"¿Qué significa la diferencia entre los dos grupos y cómo deben interpretarse los valores p y el estadístico de prueba?"
            )
            # Llamar a la IA para obtener la explicación
            respuesta = responder_pregunta_con_llama3(prompt)
            if len(respuesta) > 3000:  # Si la respuesta es muy larga, la cortamos en partes
                chunks = [respuesta[i:i + max_length] for i in range(0, len(respuesta), max_length)]
                return '\n\n'.join(chunks)  # Devolvemos las partes concatenadas
            return respuesta
    return ""


#callaback para las consultas al modelo llama3
def responder_pregunta_con_llama3(pregunta: str) -> str:
    """
    Envía la pregunta a Llama3 (Ollama) y devuelve el texto de respuesta.
    """
    try:
        # Realizar la solicitud al modelo Llama3 a través del cliente Ollama
        response = client.chat(
            model="llama3",  # Nombre del modelo
            messages=[
                {"role": "user", "content": pregunta}
            ],
            options={
                "num_predict": 1000,  # Limitar tokens de salida
                "temperature": 0.2,   # Respuestas más estables
                "max_tokens": 3000
            }
        )
        # Extraemos la respuesta generada
        content = response.get('message', {}).get('content', '').strip()
        if not content:
            raise ValueError("No se recibió una respuesta válida del modelo.")
        max_length = 3000
        if len(content) > max_length:  # Ajusta el límite según el caso
            content_parts = [content[i:i+1000] for i in range(0, len(content), max_length)]
            return '\n\n'.join(content_parts)  # Devolvemos las partes concatenadas
        return content
    except Exception as e:
        # Captura cualquier error y lo muestra
        return f"Error consultando llama3: {str(e)}"
    

# Correr la app
if __name__ == '__main__':
    app.run(debug=True)
