import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib import dates
import folium


def load_csv(filepath):
    """Load csv file and turn it to pandas dataframe

           inputs: filepath;
           returns:
               df: A  pandas dataframe
        """
    df = pd.read_csv(filepath, low_memory=False)

    return df


def clean_calendar(df):
    """Remove the dollar sign in front of price and adjusted_price, Convert the categorical varaible into binary variable,
       and turn date to datetime object.

       inputs:
           df: A pandas dataframe object
       returns:
           df: A cleaned version of dataframe
    """
    calendar = df.copy()

    calendar['price'] = calendar['price'].str.replace(',', '')
    calendar['price'] = calendar['price'].str.replace('$', '').astype(float)

    calendar['adjusted_price'] = calendar['adjusted_price'].str.replace(',', '')
    calendar['adjusted_price'] = calendar['adjusted_price'].str.replace('$', '').astype(float)


    calendar['bookings'] = calendar['available'].map(lambda x: 0 if x == 't' else 1)

    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar = calendar[calendar.date != '2020-05-05']
    calendar['month'] = calendar['date'].dt.strftime('%B')

    return calendar


def clean_reviews(viz_pack='sns'):
    """Clean the reviews dataset for easier visualization

       inputs::
            viz_pacl: string to indicate which vizualization package  to use
       returns:
           df_reviews: A cprocessed version of reviews.csv dataset based on vizualization package
        """


    df_reviews = load_csv('./data/reviews.csv.gz')

    if viz_pack == 'sns':
        df_reviews['datnum'] = dates.datestr2num(df_reviews['date'])
        df_reviews = df_reviews.groupby(['datnum'])['listing_id'].count()
        df_reviews = df_reviews.to_frame().reset_index()

    if viz_pack == 'plotly':
        df_reviews = df_reviews.groupby(['date'])['listing_id'].count()
        df_reviews = df_reviews.to_frame().reset_index()
        df_reviews['date'] = pd.to_datetime(df_reviews['date'])
        df_reviews['year'] = df_reviews['date'].dt.year
        df_reviews.index = df_reviews['date']
        df_reviews = df_reviews.rename(columns={'listing_id': 'number_of_reviews'})

    return df_reviews


def format_time(dt):
    if pd.isnull(dt):
        return "NaT"
    else:
        return datetime.strftime(dt, "%d-%b-%Y <br>")


@plt.FuncFormatter
def fake_month(x, pos):
    """ Custom formater to turn floats into Month"""
    return dates.num2date(x).strftime("%B")


@plt.FuncFormatter
def fake_year(x, pos):
    """ Custom formater to turn floats into Year"""
    return dates.num2date(x).strftime("%Y")


def review_plot(data, major_format, filename, xlabel, title, xlim=None, figsize=(8, 5)):
    """Plot the number of reviews for listings and save the figure

    Argument:
    =========
        data: pandas dataframe
        major_format: func object, formattor function
        filename: str, to save the graph
        xlabel: str
        title: str
        xlim: the x-axis ticks limits
    """

    fig, ax = plt.subplots(figsize=(8, 5))
    reviews_plot = sns.regplot(x='datnum', y='listing_id', data=data, ax=ax, lowess=True,
                               scatter_kws={"color": "#43a2ca", "alpha": 0.4},
                               line_kws={"color": "red"})
    ax.set_xlim(xlim)
    ax.xaxis.set_major_formatter(major_format)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of reviews for unqiue listings')
    fig.tight_layout()

    reviews_plot.figure.savefig(f"./images/{filename}", dpi=800);

    return reviews_plot


def find_year_maxes(df):
    """Return maximum number of reviews on each year and the date it occured

       Argument:
       =========
       df - pandas dataframes

       Return:
       =======

    """
    df = df.copy()
    df.drop(['date'], axis=1, inplace=True)
    result = pd.concat([df.groupby('year').max(),
                        df.groupby('year').idxmax()], axis=1)
    result.columns = ['number_of_reviews', 'date']

    return result.set_index('date')


def annotated_heatmap(df, col, figsize=(11, 11)):
    """Return annotated heatmap of specific columns

       Arguments
       =========
       col - a list of quantative columns
    """
    corr = df[col].corr()
    # mask to hide upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=figsize)

    heatmap = sns.heatmap(corr,
                          mask=mask,
                          square=True,
                          linewidths=.5,
                          cmap='coolwarm',
                          cbar_kws={'shrink': .5,
                                    'ticks': [-1, -.5, 0, 0.5, 1]},
                          vmin=-1,
                          vmax=1,
                          annot=True,
                          annot_kws={'size': 10})
    # add the column names as labels
    ax.set_yticklabels(corr.columns, rotation=0)
    ax.set_xticklabels(corr.columns)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})


def plot_categorical(df, x, filename, hue=None, palette='Set1', figsize=(10, 4)):
    """
    Function that plot categorical column

    Argument
    =======
    df: pandas dataframe
    x: str. horizontal axis to plot the categorical column, y would be the count
    filename: str
    hue: seaborn hue argument. Grouping variable that will produce bars with different colors.
    palette: array, color of the plot
    figsize: the size of the figure


    Returns
    =======
    A count plot for categorical column
    """
    fig, ax = plt.subplots(figsize=figsize)
    countplot = sns.countplot(x=x, hue=hue, data=df, palette=palette)
    ax.set_title(f'Count plot for {x}')
    ax.grid(False)

    countplot.figure.savefig(f'images/{filename}.png', dpi=800)


def london_map(geo_data, df, col, key_on, legend, marker=True):
    """
    Return an Interactive choropleth map of London boroughs with some of the top london attractions mark on the map

    Argument
    ========
    geo_data: geojson coordinate data
    df: pandas dataframe of London borough data
    col: the columns of data to be bound.
    key_on: Variable in the geo_data GeoJSON file to bind the data to.
    legend: str, Title for data legend.
    Returns
    ========
    folium map object

    """

    m = folium.Map(location=[51.4982, -0.1215], zoom_start=11.5)

    folium.Choropleth(
        geo_data=geo_data,
        name='choropleth',
        data=df,
        columns=col,
        key_on=key_on,
        fill_color='BuPu',
        fill_opacity=0.6,
        line_opacity=0.3,
        legend_name=legend,
    ).add_to(m)

    folium.LayerControl().add_to(m)
    if marker:
        folium.Marker([51.50777, -0.08771], popup='London Bridge').add_to(m)
        folium.Marker([51.50067, -0.12459], popup='Big Ben').add_to(m)
        folium.Marker([51.50331, -0.11965], popup='London Eye').add_to(m)
        folium.Marker([51.4993, -0.12731], popup='Westminster Abbey').add_to(m)
        folium.Marker([51.50019, -0.14237], popup='Buckingham Palace').add_to(m)
        folium.Marker([51.50441, -0.08647], popup='The Shard').add_to(m)

    return m