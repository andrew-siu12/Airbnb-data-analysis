import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_colorscales
import pandas as pd
import numpy as np
import plotly.graph_objs as go

app = dash.Dash(__name__)
server = app.server

DEFAULT_COLORSCALE = ["#2a4858", "#265465", "#1e6172", "#106e7c", "#007b84"]

BINS = ['39-57', '58-74', '75-92', '93-109', '110-149']

mapbox_access_token = 'pk.eyJ1IjoieGRnemFycSIsImEiOiJjanhrbXZpNHcyYzd2M3BsN3A3d29qbDc3In0.nSTAZlYpsFueIQLsN-hzoQ'

df_lat_lon = pd.read_csv('preprocessed_data/lat_lon_boroughs.csv')
df_reviews = pd.read_csv('preprocessed_data/prerocessed_reviews.csv')

app.css.append_css({'external_url': 'https://codepen.io/plotly/pen/EQZeaW.css'})

app.layout = html.Div(children=[
    html.Div([
        html.Div([
            html.Div([
                html.H2(children='Airbnb London Analysis',
                        style={'font-family': 'Helvetica',}
                        ),
            ]),
            html.Br(),

            html.P('Map transparency:',
                   style={
                       'display':'inline-block',
                       'verticalAlign': 'top',
                       'marginRight': '10px'
                   }
            ),

            html.Div([
                dcc.Slider(
                    id='opacity-slider',
                    min=0, max=1, value=0.8, step=0.1,
                    marks={tick: str(tick)[0:3] for tick in np.linspace(0,1,11)},
                ),
            ], style={'width':300, 'display':'inline-block', 'marginBottom':10}),

            html.Br(),

            html.Div([
                dash_colorscales.DashColorscales(
                    id='colorscale-picker',
                    colorscale=DEFAULT_COLORSCALE,
                    nSwatches=7,
                    fixSwatches=True
                )
            ], style={'display':'inline-block'}),

            html.Div([
                dcc.Checklist(
                    options=[{'label': 'Hide legend', 'value': 'hide_legend'}],
                    values=[],
                    labelStyle={'display': 'inline-block'},
                    id='hide_legend',
                    )
            ], style={'display': 'inline-block'})
        ], style={'margin':20}),

        html.P('Heatmap of London boroughs listings average price',
               id = 'heatmap-title',
               style = {'fontWeight': 500}
               ),
        html.Br(),

        dcc.Graph(
            id='boroughs-choropleth',
            figure=dict(
                data=go.Data([
                    go.Scattermapbox(
                        lat=df_lat_lon ['Latitude'],
                        lon=df_lat_lon ['Longitude'],
                        text=df_lat_lon ['Boroughs']
                    )
                ]),
                layout=dict(
                    mapbox=dict(
                        layers=[],
                        style='light',
                        accesstoken=mapbox_access_token,
                        center=dict(
                            lat=51.509865,
                            lon=-0.118092,
                        ),
                        pitch=0,
                        zoom=8
                    )
                )
            )
        ),

        html.Br(),
        html.Div([
            html.H4(children='Number of Reviews'),
            html.P('Drag the slider to change the year:'),
        ]),
        html.Br(),
        dcc.Graph(id='reviews-graphic'),

        html.Div([
            dcc.RangeSlider(
                id='years-slider',
                updatemode='mouseup',
                min=df_reviews['year'].min(),
                max=df_reviews['year'].max(),
                value=df_reviews['year'].min(),
                marks={str(year): str(year) for year in df_reviews['year'].unique()},
            ),
        ], style={'width':400, 'margin':25}),
    ], className='six columns', style={'margin':0}),
])


@app.callback(
    Output('boroughs-choropleth', 'figure'),
    [Input('opacity-slider', 'value'),
     Input('colorscale-picker', 'colorscale'),
     Input('hide_legend', 'values')],
    [State('boroughs-choropleth', 'figure')]
)
def display_map(opacity, colorscale, map_checklist, figure):
    cm = dict(zip(BINS, colorscale))

    data = [dict(
        lat=df_lat_lon['Latitude'],
        lon=df_lat_lon['Longitude'],
        text=df_lat_lon['Boroughs'],
        type='scattermapbox',
        hoverinfo='text',
        marker=dict(size=5, color='white', opacity=0)
    )]

    annotations = [dict(
        showarrow=False,
        align='right',
        text='<b>Average price of listings</b>',
        x=0.95,
        y=0.95,
    )]

    for i, bin in enumerate(reversed(BINS)):
        color = cm[bin]
        annotations.append(
            dict(
                arrowcolor=color,
                text=bin,
                x=0.95,
                y=0.85-(i/20),
                ax=-60,
                ay=0,
                arrowwidth=5,
                arrowhead=0,
                bgcolor='#EFEFEE'
            )
        )
    if 'hide_legend' in map_checklist:
        annotations = []

    if 'layout' in figure:
        lat = figure['layout']['mapbox']['center']['lat']
        lon = figure['layout']['mapbox']['center']['lon']
        zoom = figure['layout']['mapbox']['zoom']
    else:
        lat = 51.509865,
        lon = -0.118092,
        zoom = 10

    base_url = 'https://raw.githubusercontent.com/andrew-siu12/Airbnb-data-analysis/master/preprocessed_data/'

    layout = dict(
        mapbox=dict(
            layers=[dict(
                sourcetype='geojson',
                source=base_url + bin + '.geojson',
                type='fill',
                color=cm[bin],
                opacity=opacity
            ) for bin in BINS
            ],
            accesstoken=mapbox_access_token,
            style='light',
            center=dict(lat=lat, lon=lon),
            zoom=zoom
        ),
        hovermode = 'closest',
        margin = dict(r=0, l=0, t=0, b=0),
        annotations = annotations,
        dragmode = 'lasso'
    )

    fig = dict(data=data, layout=layout)
    return fig


# @app.callback(
#     Output('reviews-graphic', 'figure'),
#     [Input('xaxis-column', 'value'),
#      Input('yaxis-column', 'value'),
#      Input('xaxis-type', 'value'),
#      Input('yaxis-type', 'value'),
#      Input('years-slider', 'value')])
# def update_graph(xaxis_column_name, yaxis_column_name,
#                  xaxis_type, yaxis_type,
#                  year_value):
#     dff = df[df['Year'] == year_value]
#
#     return {
#         'data': [go.Scatter(
#             x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
#             y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
#             text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
#             mode='markers',
#             marker={
#                 'size': 15,
#                 'opacity': 0.5,
#                 'line': {'width': 0.5, 'color': 'white'}
#             }
#         )],
#         'layout': go.Layout(
#             xaxis={
#                 'title': xaxis_column_name,
#                 'type': 'linear' if xaxis_type == 'Linear' else 'log'
#             },
#             yaxis={
#                 'title': yaxis_column_name,
#                 'type': 'linear' if yaxis_type == 'Linear' else 'log'
#             },
#             margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
#             hovermode='closest'
#         )
#     }


if __name__ == '__main__':
    app.run_server(debug=True)