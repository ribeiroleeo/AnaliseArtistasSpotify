
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 21:41:24 2020

@author: leonardo
"""


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import dash_table
import xgboost as xgb


def padroniza_variavel(dados,var):

    media=np.mean(dados[var])
    dp=np.std(dados[var])
    var_pad=(dados[var]-media)/dp
    var_pad[var_pad.isna()]=0
    nome_novo="std_"+var
    dados[nome_novo]=var_pad.round(2)
    return("feito")





PAGE_SIZE = 15

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server


df = pd.read_csv("dados_musicas.csv")

variaveis=['acousticness',
       'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness',
       'speechiness', 'tempo', 'valence', 'popularity']

lista_artistas = df['artist'].unique()

app.layout = html.Div([
    html.H1(
        children='Analise Artista - Spotify',
        style={
            'textAlign': 'center'
        }
    ),
    html.Div([
    html.Div([
            html.Label("Artista 1:"),
            dcc.Dropdown(
                id='artista1Input',
                options=[{'label': i, 'value': i} for i in lista_artistas],
                value=lista_artistas[0]
                
            )
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Artista 2:"),
            dcc.Dropdown(
                id='artista2Input',
                options=[{'label': i, 'value': i} for i in lista_artistas],
                value=lista_artistas[1]
            )
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    html.Div([
        html.H3("Distribuição Popularidade"),
        html.Div([
            html.H6("Albuns-Artista1"),
            dcc.Graph(id='a1')
            ],style={'width': '30%', 'display': 'inline-block','textAlign': 'center'}),
        html.Div([
            html.H6("Artista1 vs Artista2"),
            dcc.Graph(id='a2')
            ],style={'width': '30%', 'display': 'inline-block','textAlign': 'center'}),
        html.Div([
            html.H6("Albuns-Artista2"),
            dcc.Graph(id='a3')
            ],style={'width': '30%', 'display': 'inline-block','textAlign': 'center'})
        ]),
    html.Div([
        html.H3("Análise atributos"),
        html.Div([
            html.Label("Variáveis:"),
            dcc.Dropdown(
                id='variaveisInput',
                options=[{'label': i, 'value': i} for i in variaveis],
                value=variaveis[:4],
                multi=True
            )
        ],style={'width': '100%', 'float': 'right', 'display': 'inline-block'}),
        html.Div([
            html.H6("Albuns-Artista1"),
            dcc.Graph(id='a4')
            ],style={'width': '30%', 'display': 'inline-block','textAlign': 'center'}),
        html.Div([
            html.H6("Artista1 vs Artista2"),
            dcc.Graph(id='a5')
            ],style={'width': '30%', 'display': 'inline-block','textAlign': 'center'}),
        html.Div([
            html.H6("Albuns-Artista2"),
            dcc.Graph(id='a6')
            ],style={'width': '30%', 'display': 'inline-block','textAlign': 'center'})
        ]),
    html.Div([
        html.H3("Correlação entre as variáveis"),
        html.Div([
            html.H6("Albuns-Artista1"),
            dcc.Graph(id='a7')
            ],style={'width': '30%', 'display': 'inline-block','textAlign': 'center'}),
        html.Div([
            html.H6("Geral"),
            dcc.Graph(id='a8')
            ],style={'width': '30%', 'display': 'inline-block','textAlign': 'center'}),
        html.Div([
            html.H6("Albuns-Artista2"),
            dcc.Graph(id='a9')
            ],style={'width': '30%', 'display': 'inline-block','textAlign': 'center'})
        ]),
    html.Div([
        html.H3("Classificação dos artistas"),
        html.Div([
            html.H6("Distribuição atributos entre artistas" ),
            dcc.Graph(id='a10')
            ],style={'width': '50%', 'display': 'inline-block','textAlign': 'center'}),
        html.Div([
            html.H6("Importância das variáveis na modelagem"),
            dcc.Graph(id='a11')
            ],style={'width': '50%', 'display': 'inline-block','textAlign': 'center'})]),
        html.Div([
        html.Div([
            html.H6("Diagnóstico por música"),
     
        dash_table.DataTable(
            id='datatable-paging',
    columns=[
        {"name": i, "id": i} for i in ['Musica', 'Artista', 'Artista Modelo', 'Score Artista1',
       'Score Artista2', 'std_acousticness', 'std_danceability', 'std_energy',
       'std_instrumentalness', 'std_liveness', 'std_loudness',
       'std_speechiness', 'std_tempo', 'std_valence', 'std_popularity']
    ],
    page_current=0,
    page_size=PAGE_SIZE,
    page_action='native',
    filter_action="native",
        sort_action="native",
        sort_mode="multi"
            )
            ],style={'width': '100%', 'display': 'inline-block','textAlign': 'center'})
        ])

    
])

@app.callback(
    Output('a2', 'figure'),
    [Input('artista1Input', 'value'),
     Input('artista2Input', 'value')])
def update_dist_pop_geral(artista1,artista2):
    x1 = df[df["artist"]==artista1]["popularity"]
    x2 = df[df["artist"]==artista2]["popularity"]


    hist_data = [x1, x2]

    group_labels = [artista1,artista2]

    fig=ff.create_distplot(hist_data, group_labels,
                         bin_size=4,show_hist=True,show_rug=False, 
                         colors=["lightgreen","black"])
    fig.update_layout(title_text='Distribuição de popularidade da músicas')

    return(fig)


@app.callback(
    Output('a1', 'figure'),
    [Input('artista1Input', 'value')])
def update_dist_pop_a1(artista1):
    
    dados_artista = df[df["artist"]==artista1]
    albums=dados_artista["album"].unique()
    dic={}
    for album in albums:
        dic[album]=dados_artista[dados_artista["album"]==album]["popularity"]

    l=list(dic.values())
    #group_labels = [artista1,artista2]

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(hist_data=l, group_labels=albums,curve_type="normal",
                             show_rug=False,bin_size=5,show_hist=False)


    # Add title
    fig.update_layout(title_text='Distribuição de popularidade da músicas')

    return(fig)

@app.callback(
    Output('a3', 'figure'),
    [Input('artista2Input', 'value')])
def update_dist_pop_a2(artista2):
    
    dados_artista = df[df["artist"]==artista2]
    albums=dados_artista["album"].unique()
    dic={}
    for album in albums:
        dic[album]=dados_artista[dados_artista["album"]==album]["popularity"]

    l=list(dic.values())
    #group_labels = [artista1,artista2]

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(hist_data=l, group_labels=albums,curve_type="normal",
                             show_rug=False,bin_size=5,show_hist=False)


    # Add title
    fig.update_layout(title_text='Distribuição de popularidade da músicas')

    return(fig)


@app.callback(
    Output('a7', 'figure'),
    [Input('artista1Input', 'value')])
def update_dist_cor_a1(artista1):
    

    c1 = df[df["artist"]==artista1][variaveis].corr()

    fig = go.Figure(data=go.Heatmap(
        z=c1,
        x=c1.columns,
        y=c1.index,
        colorscale=["green","white","black"]))

    fig.update_layout(title='Correlação entre as variáveis '+artista1,xaxis_nticks=36)



    return(fig)

@app.callback(
    Output('a9', 'figure'),
    [Input('artista2Input', 'value')])
def update_dist_cor_a2(artista2):
    
    c1 = df[df["artist"]==artista2][variaveis].corr()

    fig = go.Figure(data=go.Heatmap(
        z=c1,
        x=c1.columns,
        y=c1.index,
        colorscale=["green","white","black"]))

    fig.update_layout(title='Correlação entre as variáveis '+artista2,xaxis_nticks=36)



    return(fig)

@app.callback(
    Output('a8', 'figure'),
    [Input('artista1Input', 'value'),
     Input('artista2Input', 'value')])
def update_dist_cor_geral(artista1,artista2):
       

    c1 = df[variaveis].corr()

    fig = go.Figure(data=go.Heatmap(
        z=c1,
        x=c1.columns,
        y=c1.index,
        colorscale=["green","white","black"]))

    fig.update_layout(title='Correlação entre as variáveis',xaxis_nticks=36)



    return(fig)

@app.callback(
    Output('a5', 'figure'),
    [Input('artista1Input', 'value'),
     Input('artista2Input', 'value'),
     Input('variaveisInput', 'value')])
def update_dist_radar_geral(artista1,artista2,variaveis):
    if len(variaveis)<4:
        variaveis=variaveis[:4]
    
    padroniza_variavel(df,variaveis[0])
    padroniza_variavel(df,variaveis[1])
    padroniza_variavel(df,variaveis[2])
    padroniza_variavel(df,variaveis[3])



    aux=df.groupby("artist",as_index=False).agg({'std_'+variaveis[0]:np.mean,
                                                        'std_'+variaveis[1]:np.mean,
                                                        'std_'+variaveis[2]:np.mean,
                                                        'std_'+variaveis[3]:np.mean})

    v1=np.asarray(aux[aux["artist"]==artista1].iloc[0,1:])
    v1=np.concatenate((v1,[v1[0]]))
    v2=np.asarray(aux[aux["artist"]==artista2].iloc[0,1:])
    v2=np.concatenate((v2,[v2[0]]))

    variaveis_aux=variaveis[:4]
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=v1,
        theta=variaveis_aux,
        fill='toself',
        name=artista1,
        line =  dict(
                color = 'green'
                )
    
            ))
    fig.add_trace(go.Scatterpolar(
      r=v2,
      theta=variaveis_aux,
      fill='toself',
      name=artista2,
        line =  dict(
            color = 'gray'
        )
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
                )),
        showlegend=True
        )

       

    return(fig)

@app.callback(
    Output('a4', 'figure'),
    [Input('artista1Input', 'value'),
     Input('variaveisInput', 'value')])
def update_dist_radar_a1(artista1,variaveis):
    if len(variaveis)<4:
        variaveis=variaveis[:4]
    
    c1 = df[df["artist"]==artista1]

    padroniza_variavel(c1,variaveis[0])
    padroniza_variavel(c1,variaveis[1])
    padroniza_variavel(c1,variaveis[2])
    padroniza_variavel(c1,variaveis[3])



    aux=c1.groupby("album",as_index=False).agg({'std_'+variaveis[0]:np.mean,
                                                        'std_'+variaveis[1]:np.mean,
                                                        'std_'+variaveis[2]:np.mean,
                                                        'std_'+variaveis[3]:np.mean})
    
    albums=c1["album"].unique()
    #dic={}
    #colors=["green","gray"]

    #colors=sns.palplot(sns.color_palette("RdBu", n_colors=len(albums)))
    #print(albums)
    fig = go.Figure()
    for i in range(0,len(albums)):
        x=np.asarray(aux[aux["album"]==albums[i]].iloc[0,1:])
        v1=np.concatenate((x,[x[0]]))

        variaveis_aux=variaveis[:4]
        

        fig.add_trace(go.Scatterpolar(
            r=v1,
            theta=variaveis_aux,
            fill='toself',
            name=albums[i]
    
                ))
   
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
                )),
        showlegend=True
        )      

    return(fig)



@app.callback(
    Output('a6', 'figure'),
    [Input('artista2Input', 'value'),
     Input('variaveisInput', 'value')])
def update_dist_radar_a2(artista2,variaveis):
    if len(variaveis)<4:
        variaveis=variaveis[:4]
    
    c1 = df[df["artist"]==artista2]

    padroniza_variavel(c1,variaveis[0])
    padroniza_variavel(c1,variaveis[1])
    padroniza_variavel(c1,variaveis[2])
    padroniza_variavel(c1,variaveis[3])



    aux=c1.groupby("album",as_index=False).agg({'std_'+variaveis[0]:np.mean,
                                                        'std_'+variaveis[1]:np.mean,
                                                        'std_'+variaveis[2]:np.mean,
                                                        'std_'+variaveis[3]:np.mean})
    
    albums=c1["album"].unique()
    #dic={}
    #colors=["green","gray"]

    #colors=sns.palplot(sns.color_palette("RdBu", n_colors=len(albums)))
    #print(albums)
    fig = go.Figure()
    for i in range(0,len(albums)):
        x=np.asarray(aux[aux["album"]==albums[i]].iloc[0,1:])
        v1=np.concatenate((x,[x[0]]))

        variaveis_aux=variaveis[:4]
        

        fig.add_trace(go.Scatterpolar(
            r=v1,
            theta=variaveis_aux,
            fill='toself',
            name=albums[i]
    
                ))
   
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
                )),
        showlegend=True
        )      

    return(fig)

@app.callback(
    Output('a10', 'figure'),
    [Input('artista1Input', 'value'),
     Input('artista2Input', 'value')])

def update_boxplot(artista1,artista2):
    df_aux=df[df["artist"].isin([artista1,artista2])]   
    for i in variaveis:
        padroniza_variavel(df_aux,i)
        
    aux=df_aux[['std_acousticness',
                   'std_danceability',
                   'std_energy',
                   'std_instrumentalness',
                   'std_liveness',
                   'std_loudness',
                   'std_speechiness',
                   'std_tempo',
                   'std_valence',
                   'std_popularity','artist']]
    variaveis_std=['std_'+i for i in variaveis]
    aux1=pd.melt(aux,id_vars=["artist"],value_vars=variaveis_std)
    
    fig=px.box(aux1, x="variable", y="value", color="artist",color_discrete_map={artista1:"green",
                                                                                 artista2:"black"})



    return(fig)

@app.callback(
    Output('a11', 'figure'),
    [Input('artista1Input', 'value'),
     Input('artista2Input', 'value')])

def update_importancia_variaveis(artista1,artista2):
    df_aux=df[df["artist"].isin([artista1,artista2])]   

    for i in variaveis:
        padroniza_variavel(df_aux,i)
        
    variaveis_std=['std_'+i for i in variaveis]
    
    X=df_aux[variaveis_std]
    y=df_aux["artist"]
    #Treino e teste
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #print(X_train.shape,X_test.shape)
    
    #Seleção modelo
    rf=RandomForestClassifier(n_estimators=30,
                        max_depth=10)
    rf_scores = cross_validate(rf, X, y,
                        scoring='roc_auc', cv=5,
                         return_estimator=True)
    gboost=xgb.XGBClassifier(
                        n_estimators=30,
                        max_depth=10
                            )   
    gboost_scores = cross_validate(gboost, X, y,
                        scoring='roc_auc', cv=5,
                         return_estimator=True)

    rf_auc=rf_scores["test_score"].mean()
    gboost_auc=gboost_scores["test_score"].mean()

    escolha=max([rf_auc,gboost_auc])
    if escolha==rf_auc:
        titulo=("Modelo escolhido: Random Forest (AUC:{})".format(rf_auc.round(2)))
        f_imp=rf.fit(X_train,y_train).feature_importances_
        aux={"variaveis":variaveis,"importancia":f_imp}
        
        
    elif escolha==gboost_auc:
        titulo=("Modelo escolhido: XGBoost (AUC:{})".format(gboost_auc.round(2)))  
        f_imp=gboost.fit(X_train,y_train).feature_importances_
        aux={"variaveis":variaveis,"importancia":f_imp}
       
        
        
    aux1=pd.DataFrame(aux)
    aux2=aux1.sort_values(by="importancia",ascending=True)
    fig = go.Figure(go.Bar(y=aux2["variaveis"],
                           x=aux2["importancia"],   
                           orientation='h',
                           marker=dict(color='green')
                           )
                    )
    fig.update_layout(title=titulo)
    return(fig)

@app.callback(
    Output('datatable-paging', 'data'),
    [
     Input('artista1Input','value'),
     Input('artista2Input','value')])
def update_table(artista1, artista2):
    

    #Definindo modelo
    dados_aux=df[df["artist"].isin([artista1,artista2])]   

    for i in variaveis:
        padroniza_variavel(dados_aux,i)
        
    variaveis_std=['std_'+i for i in variaveis]
    
    X=dados_aux[variaveis_std]
    y=dados_aux["artist"]
    #Treino e teste
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #print(X_train.shape,X_test.shape)
    
    #Seleção modelo
    rf=RandomForestClassifier(n_estimators=30,
                        max_depth=10)
    rf_scores = cross_validate(rf, X, y,
                        scoring='roc_auc', cv=5,
                         return_estimator=True)
    gboost=xgb.XGBClassifier(
                        n_estimators=30,
                        max_depth=10
                            )   
    
    
    gboost_scores = cross_validate(gboost, X, y,
                        scoring='roc_auc', cv=5,
                         return_estimator=True)

    rf_auc=rf_scores["test_score"].mean()
    gboost_auc=gboost_scores["test_score"].mean()

    escolha=max([rf_auc,gboost_auc])
    if escolha==rf_auc:
        y_pred=rf.fit(X_train,y_train).predict(X)
        y_prob=rf.fit(X_train,y_train).predict_proba(X).round(2)
        
        
    elif escolha==gboost_auc:
        y_pred=gboost.fit(X_train,y_train).predict(X)
        y_prob=gboost.fit(X_train,y_train).predict_proba(X).round(2)
    #Tabela

    
    df_aux=pd.DataFrame({"Musica":dados_aux["name"],
                         "Artista":dados_aux["artist"],
                         "Artista Modelo":y_pred,
                         "Score Artista1":y_prob[:,0],
                         "Score Artista2":y_prob[:,1],
                         'std_acousticness':dados_aux["std_acousticness"],
                         'std_danceability':dados_aux["std_danceability"],
                         'std_energy':dados_aux["std_energy"],
                         'std_instrumentalness':dados_aux["std_instrumentalness"],
                         'std_liveness':dados_aux["std_liveness"],
                         'std_loudness':dados_aux["std_loudness"],
                         'std_speechiness':dados_aux["std_speechiness"],
                         'std_tempo':dados_aux["std_tempo"],
                         'std_valence':dados_aux["std_valence"],
                         'std_popularity':dados_aux["std_popularity"]})
    #Datatable
    return df_aux.to_dict("records")

if __name__ == '__main__':
    app.run_server(debug=True)

    
