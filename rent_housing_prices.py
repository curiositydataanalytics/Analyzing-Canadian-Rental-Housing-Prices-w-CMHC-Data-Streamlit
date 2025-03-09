# Data manipulation
import numpy as np
import datetime as dt
import pandas as pd
import geopandas as gpd

# Database and file handling
import os

# Data visualization
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from streamlit_folium import folium_static
import folium
import branca
import seaborn as sns


path_wd = os.path.dirname(os.path.realpath(__file__)).replace('\\python', '')
path_cda = '\\CuriosityDataAnalytics'
path_data = path_wd + '\\data'


# App config
#----------------------------------------------------------------------------------------------------------------------------------#
# Page config
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    .element-container {
        margin-top: -10px;
        margin-bottom: -10px;
        margin-left: -10px;
        margin-right: -10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# App title
st.title("Canadian Rental Housing Prices")
st.divider()

with st.sidebar:
    st.image(path_cda + 'logo.png')

#
#
#
#

# Loading data in cache
@st.cache_data
def load_data():
    # Census subdivisions
    CSD_2017 = gpd.read_file(path_data + '\\RMS2017_1.gdb', layer='CMHC_CSD_2017').to_crs(epsg=4326)
    CSD_2017['geometry'] = CSD_2017.geometry.centroid
    CSD_2017['lat'] = CSD_2017.geometry.y
    CSD_2017['lon'] = CSD_2017.geometry.x
    CSD_2017['CSDUID'] = CSD_2017.CSDUID.astype('int64')

    # Renting prices
    avg_rent = pd.read_csv(path_data + '\\avg_rent.csv').rename(columns={'Value' : 'avg_rent'})
    avg_rent['Date'] = pd.to_datetime(avg_rent.Date)
    avg_rent['GeoUID'] = avg_rent.GeoUID.astype('int64')
    avg_rent = avg_rent.pivot(index=['GeoUID', 'Date'], columns='Bedroom Type', values='avg_rent').add_prefix('avg_rent_').rename_axis(None, axis=1).reset_index()
    avg_rent.columns = avg_rent.columns.str.replace(' ', '_')

    # Vacancy Rates
    vcy_rate = pd.read_csv(path_data + '\\vcy_rate.csv')
    vcy_rate['Date'] = pd.to_datetime(vcy_rate.Date)
    vcy_rate['GeoUID'] = vcy_rate.GeoUID.astype('int64')
    vcy_rate = vcy_rate.pivot(index=['GeoUID', 'Date'], columns='Bedroom Type', values='vcy_rate').add_prefix('vcy_rate_').rename_axis(None, axis=1).reset_index()
    vcy_rate.columns = vcy_rate.columns.str.replace(' ', '_')
    
    # Merge metrics
    metrics = pd.merge(avg_rent, CSD_2017[['CSDUID', 'CSDNAME', 'geometry', 'lat', 'lon']], left_on='GeoUID', right_on='CSDUID', how='left')
    metrics = pd.merge(metrics, vcy_rate, on=['GeoUID', 'Date'], how='left')
    metrics = gpd.GeoDataFrame(metrics, geometry='geometry', crs=4326)
    metrics['DateStr'] = metrics.Date.astype(str)

    # Create color variables for all metrics
    def value_to_hexcolor(x, snspalette, reversed):
        if reversed:
            colorpalette=sns.color_palette(snspalette, as_cmap=True).reversed()
        else:
            colorpalette=sns.color_palette(snspalette, as_cmap=True)
        rgba = [colorpalette(plt.Normalize(x.min(), x.max())(value)) for value in x]
        hex = [mcolors.to_hex(color) for color in rgba]
        return hex
    
    metrics_cols = ['avg_rent_1_Bedroom', 'avg_rent_2_Bedroom', 'avg_rent_3_Bedroom_+', 'avg_rent_Bachelor', 'avg_rent_Total', 'vcy_rate_1_Bedroom', 'vcy_rate_2_Bedroom', 'vcy_rate_3_Bedroom_+', 'vcy_rate_Bachelor', 'vcy_rate_Total']
    for col in metrics_cols:
        if col.startswith('vcy'):
            metrics[col + '_hex'] = metrics.groupby('Date')[col].transform(value_to_hexcolor, snspalette="crest", reversed=False)
        else:
            metrics[col + '_hex'] = metrics.groupby('Date')[col].transform(value_to_hexcolor, snspalette="PuRd", reversed=False)
        metrics[col + '_opct'] = np.where(metrics[col + '_hex'] == '#000000', 0, 1)

    metrics_dict = {'avg_rent_Bachelor' : 'Average Rent - Bachelor',
                    'avg_rent_1_Bedroom' : 'Average Rent - 1 Bedroom',
                    'avg_rent_2_Bedroom' : 'Average Rent - 2 Bedrooms',
                    'avg_rent_3_Bedroom_+' : 'Average Rent -  3+ Bedrooms',
                    'avg_rent_Total' : 'Average Rent - Total',
                    'vcy_rate_Bachelor' : 'Vacancy Rate - Bachelor',
                    'vcy_rate_1_Bedroom' : 'Vacancy Rate - 1 Bedroom',
                    'vcy_rate_2_Bedroom' : 'Vacancy Rate- 2 Bedrooms',
                    'vcy_rate_3_Bedroom_+' : 'Vacancy Rate -  3+ Bedrooms',
                    'vcy_rate_Total' : 'Vacancy Rate - Total'
                    }

    return (CSD_2017, metrics, metrics_dict)
CSD_2017, metrics, metrics_dict = load_data()

#
#
#
#

# Metric selectors
cols = st.columns(2)
left_metric = {v: k for k, v in metrics_dict.items()}.get(cols[0].selectbox('Left Metric', [value for value in metrics_dict.values() if str(value).startswith('Ave')]))
right_metric = {v: k for k, v in metrics_dict.items()}.get(cols[1].selectbox('Right Metric', [value for value in metrics_dict.values() if str(value).startswith('Vaca')]))

# Date selector
cols = st.columns(1)
dt_picker = cols[0].select_slider('',value=dt.date(2023,10,1), options=metrics[metrics[left_metric].notna()].Date.dt.date.sort_values().unique())

# Create aggregated plots
cols = st.columns((0.5,0.5,0.05))
def create_plot(var):
    df = metrics.groupby('Date').agg(mean=(var, 'mean'), std=(var, 'std')).reset_index()
    df['upper'] = df['mean'] + 1.96 * df['std']
    df['lower'] = df['mean'] - 1.96 * df['std']
    fig = px.line(df, x='Date', y='mean')
    fig.add_traces([
        go.Scatter(x=df['Date'], y=df['upper'], mode='lines', line=dict(width=0), showlegend=False),
        go.Scatter(x=df['Date'], y=df['lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)', showlegend=False)
    ])
    fig.add_shape(type='line', x0=dt_picker, x1=dt_picker, y0=0, y1=df['upper'].max(), line=dict(color='rgba(173, 216, 230, 0.8)', width=2, dash='dash'))
    fig.update_xaxes(showline=True, linewidth=2, title='')
    fig.update_yaxes(showline=True, linewidth=2, title='', range=[0,df['upper'].max()])
    fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
    return fig
cols[0].plotly_chart(create_plot(left_metric), use_container_width=True)
cols[1].plotly_chart(create_plot(right_metric), use_container_width=True)

# Create map visual
m = folium.plugins.DualMap(zoom_start=3, location=(20, -97), tiles="cartodb positron")
df = metrics[metrics.Date.dt.date == dt_picker].copy()

# Legend
legendcols = st.columns((0.1,0.50,0.1))
with legendcols[1]:
    m2 = folium.Map(zoom_start=15, location=(20, -97), tiles=None, zoom_control=False, attributionControl=False)
    branca.colormap.LinearColormap(colors=[sns.color_palette('PuRd', as_cmap=True)(x) for x in np.linspace(0, 1, 1000)], vmin=df[left_metric].min(), vmax=df[left_metric].max()).add_to(m2)
    branca.colormap.LinearColormap(colors=[sns.color_palette('crest', as_cmap=True)(x) for x in np.linspace(0, 1, 1000)], vmin=df[right_metric].min(), vmax=df[right_metric].max()).add_to(m2)

    folium_static(m2, height=40, width=950)

# Rent Prices map
folium.GeoJson(df[['CSDNAME', 'geometry',left_metric, left_metric + '_hex', left_metric + '_opct']],
            marker=folium.Circle(),
            style_function=lambda x: {'fill' : True,
                                        'fillColor' : x['properties'][left_metric + '_hex'],
                                        'color': x['properties'][left_metric + '_hex'],
                                        'weight': 10,
                                        'opacity': x['properties'][left_metric + '_opct'],
                                        'fillOpacity': x['properties'][left_metric + '_opct']},
            tooltip=folium.GeoJsonTooltip(
                fields=['CSDNAME', left_metric],
                aliases=['CSD Name : ', metrics_dict.get(left_metric) + ': '],
                localize=True)
).add_to(m.m1)

# Vacancy Rate map
folium.GeoJson(df[['CSDNAME', 'geometry', right_metric, right_metric + '_hex', right_metric + '_opct']],
            marker=folium.Circle(),
            style_function=lambda x: {'fill' : True,
                                        'fillColor' : x['properties'][right_metric + '_hex'],
                                        'color': x['properties'][right_metric + '_hex'],
                                        'weight': 10,
                                        'opacity': x['properties'][right_metric + '_opct'],
                                        'fillOpacity': x['properties'][right_metric + '_opct']},
            tooltip=folium.GeoJsonTooltip(
                fields=['CSDNAME', right_metric],
                aliases=['CSD Name : ', metrics_dict.get(right_metric) + ': '],
                localize=True)
).add_to(m.m2)

# Display map
folium_static(m, height=300, width=1400)