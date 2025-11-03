import pandas as pd
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output  
import plotly.express as px
import base64
import io
from kaplan_meier import plot_kaplan_meier, plot_km_G, plot_km_disc 
#from covariables import create_abandono_graph, create_gender_graph, create_disability_graph
import numpy as np

df = pd.read_csv(r"C:\Users\lucia\Desktop\5º\TFG\prepared OULAD dataset\dataset_limpio.csv", sep= ';')
df['abandono'] = df['final_result'].apply(lambda x: 1 if x == 'Withdrawn' else 0)

df['gender'] = df['gender_F'].map({1: 'Femenino', 0: 'Masculino'})

df['disability'] = df['disability_N'].map({1: 'Con Discapacidad', 0: 'Sin Discapacidad'})


survival_analysis_page = html.Div([  
    html.H1("Survival Analysis", style={'textAlign': 'center'}),
    # Barra de navegación interna
    html.Div([
        html.Div([
            html.Img(
                src='/assets/kaplan.png', 
                style={'width': '60%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px','marginBottom': '15px'}
            ),
            dcc.Link('Kaplan-Meier', href='/survival-analysis/kaplan-meier', className='home-link'),

        ], style={'textAlign': 'center',  'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'marginTop': '20px', 'width': '30%'}),

        html.Div([
            html.Img(
                src='/assets/cox.png', 
                style={'width': '60%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px','marginBottom': '30px'}  
            ),
            dcc.Link('Cox Regression', href='/survival-analysis/cox-regression', className='home-link'),
        ], style={'textAlign': 'center',  'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'marginTop': '20px', 'width': '30%'}),  

        html.Div([
            html.Img(
                src='/assets/logrank.png', 
                style={'width': '50%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px','marginBottom': '30px'} 
            ),
            dcc.Link('Log-Rank Test', href='/survival-analysis/log-rank', className='home-link'),
        ], style={'textAlign': 'center',  'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'marginTop': '20px', 'width': '30%'}),  
    ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center', 'gap': '30px', 'marginTop': '20px'}),  

    # Contenedor para mostrar el gráfico de Kaplan-Meier
    html.Div(id='survival_analysis_page', children=[]),
        # Este div se llenará con el gráfico de Kaplan-Meier cuando se acceda a la ruta correspondiente
        
])

ver_dataset_page = html.Div([ 
    html.H1("Dataset Limpio", style={'textAlign': 'center', 'fontSize': '2.5em'}),
    html.Div([ 
        dash_table.DataTable(
            id='clean-dataset-table',
            columns=[{"name": col, "id": col} for col in df.columns],  
            data=df.to_dict('records'),  # convertir el DataFrame en un formato que Dash pueda usar
            style_table={'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '15px'},  # Asegurarse de que el texto esté alineado
        ),
    ], style={'textAlign': 'center', 'marginTop': '30px'})
])

# Página de análisis de covariables
covariate_analysis_page = html.Div([
    html.Div(className='row', children=[
        html.H1("Análisis de Covariables ", style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center', 'gap': '30px', 'marginTop': '20px'}),
    ]),
    html.Div(className='row', children=[
        dcc.RadioItems(
            options=[
                {'label': 'Abandono Total', 'value': 'abandono'},
                {'label': 'Abandono por Género', 'value': 'gender'},
                {'label': 'Abandono por Discapacidad', 'value': 'disability'}
            ],
            value='abandono',
            id='covariables-dropdown',
            style={'width': '80%', 'padding': '10px', 'margin': 'auto', 'display': 'block'}
        )
    ], style={'textAlign': 'center', 'marginTop': '30px'}),
    
    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
            dcc.Graph(id='covariables-graph')  
        ], style={'width': '70%', 'padding': '10px', 'display': 'inline-block'}),  

        html.Div(className='six columns', children=[  
            html.Div(id='graph-explanation', style={'padding': '10px', 'display': 'flex', 'alignItems': 'center'})
        ],style={
            'position': 'fixed',        #  barra en el lado derecho
            'top': '7%',               
            'right': '0',               # Pegado al margen derecho
            'z-index': '1000',          # se superponga sobre otros elementos
            'backgroundColor': '#f4f7f6',  # fondo diferenciado de la página
            'width': '300px',           # ancho de la barra lateral
            'height': '100vh',  # la barra ocupe todo el alto de la pantalla
            'borderLeft': '2px solid #ccc',  # borde para diferenciarla
            'padding': '10px',
            'display': 'flex', 
            'color': '#6A6A6A',
            'flexDirection': 'column',  # contenido dentro centrado
            'justifyContent': 'center'
        })
    ], style={'display': 'flex', 'alignItems': 'center'})
])

#Pagina de regresion de Cox
cox_regression_page = html.Div([
    html.H1("Survival Analysis: Regresión de Cox", style={'textAlign': 'center','fontSize': '35px'}),
    
    html.Div([
        #html.Label("Selecciona 1 o más covariable para la regresión de Cox:", style={'fontWeight':'bold'}),
        dcc.Dropdown(
            id='covariables-dropdown-cox',
            options=[
                {'label': 'Género', 'value': 'gender_F'},
                {'label': 'Discapacidad', 'value': 'disability_N'}],
            placeholder='Elige 1 o más covariable para la regresión de Cox',  
            multi=True,  # Permite seleccionar varias covariables
            style={'width': '60%', 'margin': 'auto'}
        ),
    ], style={'textAlign': 'center', 'marginTop': '30px'}),
    
    html.Div(id='cox-regression-output', style={'textAlign': 'center', 'marginTop': '20px'}),
    html.Button('Explicar', id='btn-cox', style={'margin-top': '20px', 'display': 'block', 'margin': 'auto'}),
    html.Div(
        dcc.Textarea(
            id='openai-answer-cox',
            placeholder='La respuesta es...',
            style={
                'width': '60%',  #  tamaño
                'height': '400px',  # altura de la caja de texto
                'resize': 'none',  
                'whiteSpace': 'pre-wrap',  # texto se ajuste y no se corte
                'margin': '20px auto',  # centrado automático
                'display': 'block',  # en bloque y centrado
                'border': '1px solid #ccc',  # borde para darle un poco de estilo
                'borderRadius': '8px',  # bordes redondeados
                'fontSize': '16px',  #  tamaño de la fuente para mayor legibilidad
                'overflowY': 'auto'
            },
            disabled=True 
        ),
        style={'textAlign': 'center'}  # centrado del contenedor que envuelve el Textarea
    ) 
])

# Página de Kaplan–Meier
kaplan_meier_page = html.Div([
    html.H1("Survival Analysis: Kaplan-Meier", style={'textAlign': 'center', 'fontSize': '35px'}),
    
    # 1) Gráfica global
    html.Div(id='km-global-div', children=plot_kaplan_meier(df)),
    
    html.Div([
        html.Label("Selecciona 1 covariable para ver su curva de Kaplan:", style={'fontWeight':'bold'}),
        html.Div([
            html.Button('Género', id='botonG', style={'border-radius': '50%', 'padding': '10px 20px', 'margin': '10px'}),
            html.Button('Discapacidad', id='botonDisc', style={'border-radius': '50%', 'padding': '10px 20px', 'margin': '10px'}),
            html.Button('Ninguna', id='botonNone', style={'border-radius': '50%', 'padding': '10px 20px', 'margin': '10px', 'background-color': '#accfc3'}),

        ], style={'textAlign': 'center'}),
    ], style={'textAlign': 'center','marginTop':'30px'}),


    # 3) Div donde irá la gráfica de la covariable
    html.Div(id='km-cov-div', style={'marginTop':'20px'}),
    html.Button('Explicar', id='explicar-btn-kaplan', style={'margin-top': '20px', 'display': 'block', 'margin': 'auto'}),

    html.Div(
        dcc.Textarea(
            id='openai-answer-kaplan',
            placeholder='La respuesta es...',
            style={
                'width': '60%',  
                'height': '400px', 
                'resize': 'none',  
                'whiteSpace': 'pre-wrap', 
                'margin': '20px auto',  
                'display': 'block',  
                'border': '1px solid #ccc', 
                'borderRadius': '8px',  
                'fontSize': '16px', 
                'overflowY': 'auto'
            },
            disabled=True  
        ),
        style={'textAlign': 'center'}
    )
])

#pagina de analisis de log-rank
log_rank_page=html.Div([
    html.H1("Survival Analysis: Test de Log-Rank", style={'textAlign': 'center', 'fontSize': '35px'}),
    
    html.Div([
        #html.Label("Selecciona 1 o más covariables para el Test de Log-Rank:", style={'fontWeight':'bold'}),
        dcc.Dropdown(
            id='covariables-dropdown-logrank',
            options=[
                {'label': 'Género', 'value': 'gender_F'},
                {'label': 'Discapacidad', 'value': 'disability_N'},
            ],
            placeholder='Selecciona 1 o más covariables para el Test de Log-Rank:',  
            value=[], 
            multi= True,
            style={'width': '60%', 'margin': 'auto'}
        ),
    ], style={'textAlign': 'center', 'marginTop': '30px'}),

    # Este div se actualizará con el resultado del Log-Rank Test
    html.Div(id='logrank-test-output', style={'textAlign': 'center', 'marginTop': '20px'}),
    html.Button('Explicar', id='explicar-btn-logrank', style={'margin-top': '20px', 'display': 'block', 'margin': 'auto'}),

    html.Div(
        dcc.Textarea(
            id='openai-answer-logrank',
            placeholder='La respuesta es...',
            style={
                'width': '60%',  
                'height': '400px',  
                'resize': 'none',  
                'whiteSpace': 'pre-wrap',  
                'margin': '20px auto', 
                'display': 'block',  
                'border': '1px solid #ccc',
                'borderRadius': '8px',  
                'fontSize': '16px', 
                'overflowY': 'auto'
            },
            disabled=True  
        ),
        style={'textAlign': 'center'} 
    )
])

def display_logrank_summary_table(result):
    df_show = result[['Covariable','Grupo A','Grupo B',
                      'n A','n B','test_statistic','p_value','-log2(p)',
                      'Decisión','Conclusión']].copy()

    return dash_table.DataTable(
        id='logrank-summary-table',
        columns=[{"name": col, "id": col} for col in df_show.columns],
        data=df_show.to_dict('records'),
        style_table={'height': '150px', 'overflowY': 'auto', 'width': '58%', 'margin': 'auto'},
        style_cell={'textAlign': 'center', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '15px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f4f7f6'},
        style_data_conditional=[
            {'if': {'filter_query': '{decision} = "Rechazar H0"'}, 'backgroundColor': '#ffecec'},
            {'if': {'filter_query': '{decision} = "No rechazar H0"'}, 'backgroundColor': '#ecffec'},
        ]
    )

