import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import networkx as nx
from matplotlib import cm
import matplotlib.colors as colors
from sklearn.cluster import DBSCAN
from wordcloud import WordCloud


# functions
def book_average_rating(reviews):
    total_reviews = sum(reviews)
    weighted_sum = sum((reviews.index(count) + 1) * count for count in reviews)

    if total_reviews == 0:
        return 0

    average_rating = round(weighted_sum / total_reviews, 2)
    return average_rating

def plot_network(G, title, scale_legend) :

    pos = nx.layout.spring_layout(G)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='grey'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=False,
            color=[],
            size=15,
            colorbar=dict(
                thickness=15,
                title= scale_legend,
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    node_adjacencies = []
    node_text = []
    label = 0
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(str(list(G.nodes())[label]) + " " + scale_legend + " : " + str(len(adjacencies[1])))
        label += 1

    #node_trace.marker.color = node_adjacencies
    node_colors = ['orange' if str(list(G.nodes())[i])[0] == 'u' else node_adjacencies[i] for i in range(len(list(G.nodes())))]
    node_trace.marker.color = node_colors
    node_trace.text = node_text

    return go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=title,
                    plot_bgcolor='rgba(0,0,0,0)',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002)],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

def plot_network_clusters(G, title, scale_legend) :

    pos = nx.layout.spring_layout(G)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='grey'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=False,
            color=[],
            size=15,
            colorbar=dict(
                thickness=15,
                title= scale_legend,
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    partition = {node: community_id for community_id, community in
                 enumerate(nx.community.louvain_communities(G, seed=123)) for node in community}
    node_colors_community = [partition[node] for node in G.nodes()]

    node_adjacencies = []
    node_text = []
    label = 0
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(str(list(G.nodes())[label]) + " " + scale_legend + " : " + str(partition[list(G.nodes())[label]]))
        label += 1

    node_trace.marker.color = node_colors_community
    node_trace.text = node_text

    return go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=title,
                    plot_bgcolor='rgba(0,0,0,0)',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002)],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

def plot_network_centrality(G, title, centrality_measure='Degré', node_size_multiplier=1.0):
    pos = nx.layout.spring_layout(G)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='grey'),
        hoverinfo='none',
        mode='lines'
    )

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Reds',
            reversescale=False,
            color=[],
            size=15 * node_size_multiplier,
            colorbar=dict(
                thickness=15,
                title= str(centrality_measure),
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)
        )
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    if centrality_measure == 'Degré':
        centrality_values = dict(G.degree())
    elif centrality_measure == 'Betweenness':
        centrality_values = nx.betweenness_centrality(G)
    elif centrality_measure == 'Closeness':
        centrality_values = nx.closeness_centrality(G)
    elif centrality_measure == 'Vecteur Propre':
        centrality_values = nx.eigenvector_centrality(G)
    else:
        raise ValueError("Invalid centrality measure. Choose from 'degree', 'betweenness', 'closeness', 'farness', or 'eigenvector'.")

    node_colors_centrality = [centrality_values[node] for node in G.nodes()]
    node_trace.marker.color = node_colors_centrality

    node_text = [f"{list(G.nodes())[label]} {centrality_measure.capitalize()}: {centrality_values[node]:.3f}" for label, node in enumerate(G.nodes())]

    node_trace.text = node_text

    return go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         title=title,
                         plot_bgcolor='rgba(0,0,0,0)',
                         titlefont=dict(size=16),
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20, l=5, r=5, t=40),
                         annotations=[dict(
                             showarrow=False,
                             xref="paper", yref="paper",
                             x=0.005, y=-0.002)],
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                     )
                     )

# load the data
data = pd.read_csv('PROJECT_DA_books.csv', encoding='utf-8')
data = data[data['original_publication_year'] != 0]
data = data.reset_index(drop=True)

# genres
genres = [j for i in data['genres'] for j in eval(i)]
all_genres = list(set(genres))
genre_counts = pd.Series(genres).value_counts(normalize=True) * 100

# authors
authors = [j for i in data['authors'] for j in eval(i)]
all_authors = list(set(authors))
author_counts = pd.Series(authors).value_counts()

# avg ratings by year of publication
ratings = [[data['ratings_1'][i], data['ratings_2'][i],
               data['ratings_3'][i], data['ratings_4'][i],
               data['ratings_5'][i]] for i in range(len(data))]

avg_ratings = [book_average_rating(i) for i in ratings]
pub_year = data['original_publication_year']

# ratings
title_options = data['title'].apply(lambda title: {'label': title, 'value': title}).to_list()

# description
description_emb_x = data['desc_embeddings_x']
description_emb_y = data['desc_embeddings_y']

dbscan = DBSCAN(eps=3, min_samples=5, metric='euclidean')
data['cluster_desc'] = dbscan.fit_predict(np.column_stack((description_emb_x, description_emb_y)))

# book covers
cover_emb_x = data['img_embeddings_x']
cover_emb_y = data['img_embeddings_y']

dbscan = DBSCAN(eps=1.4, min_samples=5, metric='euclidean')
data['cluster_cover'] = dbscan.fit_predict(np.column_stack((cover_emb_x, cover_emb_y)))

# users ratings
df_ratings = pd.read_csv("ratings.csv")
df_ratings['user_id'] = df_ratings['user_id'].apply(lambda x: 'u_' + str(x))

books_rated_counts = df_ratings['book_id'].value_counts()
books_rated_counts = books_rated_counts[0:10]
id_to_title_mapping_ratings = data.set_index('book_id')['title'].to_dict()
books_rated_counts.index = [id_to_title_mapping_ratings[i] for i in books_rated_counts.index]

df_ratings = df_ratings[0:500]
tuple_list_ratings = [tuple(row[:-1]) for row in df_ratings.to_numpy()]

G_ratings = nx.Graph()
G_ratings.add_edges_from(tuple_list_ratings)

fig_ratings_network = plot_network(G_ratings, "Réseau des notations des ouvrages par les utilisateurs", "Nombre de notes attribuées")
fig_ratings_network_clusters = plot_network_clusters(G_ratings, "Clustering des notations des ouvrages par les utilisateurs", "Cluster")


# users to_read
df_to_read = pd.read_csv("to_read.csv")
df_to_read['user_id'] = df_to_read['user_id'].apply(lambda x: 'u_' + str(x))

books_to_read_counts = df_to_read['book_id'].value_counts()
books_to_read_counts = books_to_read_counts[0:10]
id_to_title_mapping_to_read = data.set_index('book_id')['title'].to_dict()
books_to_read_counts.index = [id_to_title_mapping_to_read[i] for i in books_to_read_counts.index]

df_to_read = df_to_read[0:500]
tuple_list_to_read = [tuple(row) for row in df_to_read.to_numpy()]

G_to_read = nx.Graph()
G_to_read.add_edges_from(tuple_list_to_read)

fig_to_read_network = plot_network(G_to_read, "Réseau des ouvrages 'à lire'", "Nombre de mentions 'à lire'")
fig_to_read_network_clusters = plot_network_clusters(G_to_read, "Clustering des ouvrages 'à lire'", "Cluster")

# tags
df_tags = pd.read_csv("tags.csv")
text = ' '.join(df_tags['tag_name'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)


app = dash.Dash(__name__, external_stylesheets=['styles.css'])

app.layout = html.Div(children=[
    html.Div(className='title-container', children=[
    html.H1(className='title', children='📚 Analyse de données du dataset 10k goodreads'),
    ]),
    html.Div(className='container', children=[
html.Div(className='subtitle-container', children=[
    html.H1(className='subtitle', children='Données des ouvrages :'),
    ]),
    html.Div(className='container1', children=[
        dcc.Graph(
            id='authors-bar-chart',
            figure={
                'data': [
                    {
                        'x': list(author_counts.index)[0:20],
                        'y': author_counts[0:20],
                        'type': 'bar',
                        'textposition': 'auto',
                        'marker': {
                            'color': author_counts[0:20],  # Use counts as color values
                            'colorscale': 'Viridis',  # You can choose a different color scale
                            'showscale': False,
                        },
                    }
                ],
                'layout': {
                    'title': "Top 20 des auteurs par nombre des ouvrages",
                    'xaxis': {'title': 'Auteurs'},
                    'yaxis': {'title': "Nombre des ouvrages"},
'margin': {'b': 200},
'font': {'family': 'Outfit, sans-serif'}
                }
            }
        ),

        dcc.Graph(
        id='genre-bar-chart',
        figure={
            'data': [
                {
                    'x': list(genre_counts.index),
                    'y': genre_counts,
                    'type': 'bar',
                    'textposition': 'auto',
'marker': {
                            'color': genre_counts,  # Use counts as color values
                            'colorscale': 'Viridis',  # You can choose a different color scale
                            'showscale': False,
                        },
                }
            ],
            'layout': {
                'title': "Distribution des genres des ouvrages",
                'xaxis': {'title': 'Genres'},
                'yaxis': {'title': "Pourcentage des ouvrages"},
'margin': {'b': 200},
'font': {'family': 'Outfit, sans-serif'}
            }
        }
    ),
dcc.Graph(
        id='avg-ratings-scatter-chart',
        figure={
            'data': [
                {
                    'x': pub_year,
                    'y': avg_ratings,
                    'type': 'scatter',
                    'hovertext': data['title'],
                    'textposition': 'auto',
                    'mode': 'markers',
'marker': {
                            'color': avg_ratings,  # Use counts as color values
                            'colorscale': 'Viridis',  # You can choose a different color scale
                            'showscale': False,
                        },
                }
            ],
            'layout': {
                'title': "Distribution des moyennes des notes par année de publication",
                'xaxis': {'title': 'Années de publication', 'zeroline': False},
                'yaxis': {'title': "Notes moyenne des ouvrages", 'zeroline': False},
'font': {'family': 'Outfit, sans-serif'}
            }
        }
    ),
    html.Div(className='dropdown-container', children=[
    dcc.Dropdown(
        id='dropdown-menu',
        options=title_options,
        value="The Hunger Games (The Hunger Games, #1)",
        style={'width': '80%'}
    ),

    dcc.Graph(id='scatter-plot'),
        ]),
]),
html.Div(className='subtitle-container', children=[
    html.H1(className='subtitle', children='Clustering des embeddings :'),
    ]),
html.Div(className='container2', children=[
dcc.Graph(
        id='desc-clustering-scatter-chart',
        figure={
            'data': [
                {
                    'x': description_emb_x,
                    'y': description_emb_y,
                    'type': 'scatter',
                    'marker': {'color': data['cluster_desc'], 'opacity': 0.4},
                    'hovertext': data['title'],
                    'textposition': 'auto',
                    'mode': 'markers',

                }
            ],
            'layout': {
                'title': "Clustering des ouvrages suivant leurs descriptions",
                'xaxis': {'title': 'dim_1', 'zeroline': False},
                'yaxis': {'title': "dim_2", 'zeroline': False},
'font': {'family': 'Outfit, sans-serif'}
            }
        }
    ),
dcc.Graph(
        id='cover-clustering-scatter-chart',
        figure={
            'data': [
                {
                    'x': cover_emb_x,
                    'y': cover_emb_y,
                    'type': 'scatter',
                    'marker': {'color': data['cluster_cover'], 'opacity': 0.4},
                    'hovertext': data['title'],
                    'textposition': 'auto',
                    'mode': 'markers',
                }
            ],
            'layout': {
                'title': "Clustering des ouvrages suivant leurs première page de couverture",
                'xaxis': {'title': 'dim_1', 'zeroline': False},
                'yaxis': {'title': "dim_2", 'zeroline': False},
'font': {'family': 'Outfit, sans-serif'}
            }
        }
    ),
]),
html.Div(className='subtitle-container', children=[
    html.H1(className='subtitle', children="Données des ouvrages 'à lire' et des ouvrages notés par les utilisateurs:"),
    ]),
html.Div(className='container3', children=[
dcc.Graph(
        id='ratings-books-bar-chart',
        figure={
            'data': [
                {
                    'x': list(books_rated_counts[0:10].index),
                    'y': books_rated_counts[0:10],
                    'type': 'bar',
                    'textposition': 'auto',
'marker': {
                            'color': author_counts[0:20],  # Use counts as color values
                            'colorscale': 'Viridis',  # You can choose a different color scale
                            'showscale': False,
                        },
                }
            ],
            'layout': {
                'title': "Top 10 des ouvrages selon leur nombre de notation",
                'xaxis': {'title': 'Ouvrages'},
                'yaxis': {'title': "Nombre de mention 'à lire'"},
'margin': {'b': 200},
'font': {'family': 'Outfit, sans-serif'}
            }
        }
    ),
dcc.Graph(
        id='ratings-users-bar-chart',
        figure={
            'data': [
                {
                    'x': list(df_ratings['user_id'].value_counts()[0:20].index),
                    'y': df_ratings['user_id'].value_counts()[0:20],
                    'type': 'bar',
                    'textposition': 'auto',
'marker': {
                            'color': author_counts[0:20],  # Use counts as color values
                            'colorscale': 'Viridis',  # You can choose a different color scale
                            'showscale': False,
                        },
                }
            ],
            'layout': {
                'title': "Top 20 des utilisateurs selon leur nombre des ouvrages notés",
                'xaxis': {'title': 'Ouvrages'},
                'yaxis': {'title': "Nombre de mention 'à lire'"},
'margin': {'b': 200},
'font': {'family': 'Outfit, sans-serif'}
            }
        }
    ),
dcc.Graph(
        id='to-read-books-bar-chart',
        figure={
            'data': [
                {
                    'x': list(books_to_read_counts[0:10].index),
                    'y': books_to_read_counts[0:10],
                    'type': 'bar',
                    'textposition': 'auto',
'marker': {
                            'color': author_counts[0:20],  # Use counts as color values
                            'colorscale': 'Viridis',  # You can choose a different color scale
                            'showscale': False,
                        },
                }
            ],
            'layout': {
                'title': "Top 10 des ouvrages 'à lire'",
                'xaxis': {'title': 'Ouvrages'},
                'yaxis': {'title': "Nombre de mention 'à lire'"},
'margin': {'b': 200},
'font': {'family': 'Outfit, sans-serif'}
            }
        }
    ),
dcc.Graph(
        id='to-read-users-bar-chart',
        figure={
            'data': [
                {
                    'x': list(df_to_read['user_id'].value_counts()[0:20].index),
                    'y': df_to_read['user_id'].value_counts()[0:20],
                    'type': 'bar',
                    'textposition': 'auto',
'marker': {
                            'color': author_counts[0:20],  # Use counts as color values
                            'colorscale': 'Viridis',  # You can choose a different color scale
                            'showscale': False,
                        },
                }
            ],
            'layout': {
                'title': "Top 20 des utilisateurs selon leurs nombres des ouvrages 'à lire'",
                'xaxis': {'title': 'Utilisateurs'},
                'yaxis': {'title': "Nombre des ouvrages 'à lire'"},
'margin': {'b': 200},
'font': {'family': 'Outfit, sans-serif'}
            }
        }
    ),
]),
html.Div(className='subtitle-container', children=[
    html.H1(className='subtitle', children='Analyse des graphes :'),
    ]),
html.Div(className='container4', children=[
html.Div(dcc.Graph(id='fig_ratings_network', figure=fig_ratings_network)),
html.Div(dcc.Graph(id='fig_ratings_network_clusters', figure=fig_ratings_network_clusters)),
html.Div(dcc.Graph(id='fig_to_read_network', figure=fig_to_read_network)),
html.Div(dcc.Graph(id='fig_to_read_network_clusters', figure=fig_to_read_network_clusters)),
html.Div(className='dropdown-container2', children=[
dcc.Dropdown(
        id='to_read_centrality_dropdown',
        options=[
            {'label': 'Degree Centrality', 'value': 'Degré'},
            {'label': 'Betweenness Centrality', 'value': 'Betweenness'},
            {'label': 'Closeness Centrality', 'value': 'Closeness'},
            {'label': 'Eigenvector Centrality', 'value': 'Vecteur Propre'},
        ],
        value='Degré',
        style={'width': '60%'}
    ),
html.Div(dcc.Graph(id='fig_to_read_network_measures')),
]),
html.Div(className='dropdown-container2', children=[
dcc.Dropdown(
        id='ratings_centrality-dropdown',
        options=[
            {'label': 'Degree Centrality', 'value': 'Degré'},
            {'label': 'Betweenness Centrality', 'value': 'Betweenness'},
            {'label': 'Closeness Centrality', 'value': 'Closeness'},
            {'label': 'Eigenvector Centrality', 'value': 'Vecteur Propre'},
        ],
        value='Degré',
        style={'width': '60%'}
    ),
html.Div(dcc.Graph(id='fig_ratings_network_measures'))
]),
]),
]),
])


@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('dropdown-menu', 'value')]
)
def update_ratings_plot(title):
    book_index = list(data['title'][data['title'] == title].index)[0]
    ratings = [data['ratings_1'][book_index], data['ratings_2'][book_index],
               data['ratings_3'][book_index], data['ratings_4'][book_index],
               data['ratings_5'][book_index]]

    fig = go.Figure(data=[
        go.Bar(
            x=['★', '★★', '★★★', '★★★★', '★★★★★'],
            y=ratings,
            textposition='auto',
            marker=dict(
                color=ratings,
                colorscale='Viridis',
                showscale=False,
            ),
        )
    ])

    fig.update_layout(
        title="Distribution des notes des ouvrages",
        xaxis_title='Notes',
        yaxis_title="Nombre de notes",
        bargap=0.6,
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

@app.callback(
    Output('fig_ratings_network_measures', 'figure'),
    [Input('ratings_centrality-dropdown', 'value')]
)
def update_graph(selected_centrality):
    fig = plot_network_centrality(G_ratings, 'Mesures du réseau des notations des ouvrages par les utilisateurs', centrality_measure=selected_centrality, node_size_multiplier=1.5)
    return fig

@app.callback(
    Output('fig_to_read_network_measures', 'figure'),
    [Input('to_read_centrality_dropdown', 'value')]
)
def update_graph(selected_centrality):
    fig = plot_network_centrality(G_to_read, "Mesures du réseau des ouvrages 'à lire'",centrality_measure=selected_centrality, node_size_multiplier=1.5)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
