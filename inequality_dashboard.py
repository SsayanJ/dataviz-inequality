# -*- coding: utf-8 -*-
from enum import unique
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
from dash_core_components.Dropdown import Dropdown
from dash_core_components.Markdown import Markdown
import dash_html_components as html
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Get unique values in a series dropping NaN values


def unique_non_null(s):
    return s.dropna().unique()


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

##### UTILS FUNCTIONS #####

# Overall period is 60 years so divisors of 60 to be preferred for period


def indicator_period_mean(dataframe, indicator, period):
    df1 = dataframe.drop(dataframe[dataframe['Year'] == 2020].index)
    df1 = df1.groupby(np.arange(len(df1.index))//period).agg({indicator: 'mean', 'Year': [
        'first', 'last'], "Country Name": "first", "Region": "first", "IncomeGroup": "first"})
    df1.columns = df1.columns.map('_'.join)
    df1.insert(0, 'Years', df1['Year_first'].astype(
        str) + '-' + df1['Year_last'].astype(str))
    df1.drop(['Year_first', 'Year_last'], axis=1, inplace=True)
    df1.columns = ["Period", indicator,
                   "Country Name", "Region", "IncomeGroup"]
    return df1


def mean_per_country(dataframe):
    df2 = dataframe.groupby("Country Name").mean().reset_index()
    df2 = df2.drop(columns='Year')
    df2 = pd.merge(df2, all_meta, left_on='Country Name', right_on='TableName',
                   how='left').drop(columns=['TableName', 'Unnamed: 5', 'SpecialNotes'])

    df2 = pd.merge(df2, democracy_index.reset_index()[
                   ['Country', 'Regime type']], how='left', left_on='Country Name', right_on="Country")
    df2 = df2.rename(columns={"Regime type": "Regime Type"}, errors='raise')
    return df2


def mean_per_region(dataframe):
    df2 = dataframe.groupby("Region").mean().reset_index()
    df2 = df2.drop(columns='Year')
    return df2


def mean_per_incomegroup(dataframe):
    df2 = dataframe.groupby("IncomeGroup").mean().reset_index()
    df2 = df2.drop(columns='Year')
    return df2


def mean_per_regime(dataframe):
    df2 = dataframe.groupby("Regime Type").mean().reset_index()
    df2 = df2.drop(columns='Year')
    return df2


def mean_per_year(dataframe):
    df2 = dataframe.groupby("Year").mean().reset_index()
    return df2

###### DATA PROCESSING ######


gini = pd.read_csv("data/WIID_19Dec2018.csv", delimiter=';')
all_meta = pd.read_csv("data/Metadata_Country_gini.csv")
country_meta = all_meta.dropna(subset=['Region'])

# Merge the Region and Income level with Gini score
gini_meta = pd.merge(gini, country_meta, on='Country Code')

# Transform the "pivot table" DF style into a "database" type
country_gini = gini.drop(columns=[
                         'Indicator Name', 'Indicator Code', 'Unnamed: 65']).set_index("Country Name")
transform_list = []
for c in country_gini.index:
    for y in [str(y) for y in range(1960, 2021)]:
        transform_list.append(
            [c, country_gini.loc[c, 'Country Code'], int(y), country_gini.loc[c, y]])

gini_db = pd.DataFrame(transform_list, columns=[
                       'Country Name', 'Country Code', 'Year', 'Gini Coeff'])

# Add Region and Income Level. This "main_db" will be the main DB where we add all other indicators per country and per year
main_db = pd.merge(gini_db, all_meta, on='Country Code').drop(
    columns=['TableName', 'Unnamed: 5'])

# Removing years and countries with no data
gini_meta = gini_meta.dropna(
    how='all', subset=[str(i) for i in range(1960, 2020)])
gini_meta = gini_meta.dropna(axis=1, how='all')


gini_countries = gini_meta.set_index("Country Name")
gini_countries = gini_countries.drop(
    ["Country Code", "Indicator Name", "Indicator Code", "Region", "IncomeGroup", "SpecialNotes", "TableName"], axis=1)
gini_mean = gini_countries.mean(axis=0)


##### Transparency Data Processing #####
transparency = pd.read_csv(
    "data/transparency_score_2012-2020.csv", delimiter=';').set_index("Country")
transform_transparency = []
for c in transparency.index:
    for y in range(2012, 2021):
        transform_transparency.append(
            [c, transparency.loc[c, 'ISO3'], y, transparency.loc[c, f"CPI score {y}"]])

transparency_db = pd.DataFrame(transform_transparency, columns=[
                               'Country Name', 'Country Code', 'Year', 'Transparency Score'])

# Merge transparency data with main_db
main_db = pd.merge(main_db, transparency_db[['Country Code', 'Year', 'Transparency Score']], on=[
                   'Country Code', 'Year'], how='left')

##### GDP per capita Data Processing #####
gdp = pd.read_csv("data/WID_GDP_per_capita.csv").set_index("Country Name")

transform_gdp = []
for c in gdp.index:
    for y in range(1960, 2021):
        gdp_y = gdp.loc[c, str(y)]
        transform_gdp.append(
            [gdp.loc[c, 'Country Code'], y, gdp_y if pd.isna(gdp_y) else int(gdp_y)])

gdp_db = pd.DataFrame(transform_gdp, columns=[
    'Country Code', 'Year', 'GDP per capita'])
# Merge transparency data with main_db
main_db = pd.merge(main_db, gdp_db, on=[
                   'Country Code', 'Year'], how='left')

##### Democracy index Processing #####
democracy_index = pd.read_csv(
    "data/Wikipedia-democracy-index(eiu).csv", delimiter=";").set_index("Country")
transform_democracy = []
years = list(range(2010, 2021))
years.extend([2006, 2008])
for c in democracy_index.index:
    for y in years:
        transform_democracy.append(
            [c, y, democracy_index.loc[c, str(y)], democracy_index.loc[c, 'Regime type']])
democracy_index_db = pd.DataFrame(transform_democracy, columns=[
                                  'Country Name', 'Year', 'Democracy Index', 'Regime Type'])

# Merge democracy index with main_db
main_db = pd.merge(main_db, democracy_index_db,
                   how='left', on=['Country Name', 'Year'])


mean_db = mean_per_country(main_db)
incomegroup_db = mean_per_incomegroup(main_db)
region_db = mean_per_region(main_db)

main_db_2010_2019 = main_db[main_db['Year'] >= 2010]
main_db_2010_2019 = main_db_2010_2019[main_db_2010_2019['Year'] < 2020]
region_db_2010_2019 = mean_per_region(main_db_2010_2019)

##### Graph List #####
gini_graph = px.scatter(mean_per_year(main_db).dropna(
    subset=['Gini Coeff']), x="Year", y="Gini Coeff")
gini_transparency_corr = px.scatter(main_db[main_db["Year"] == 2018].dropna(subset=['Gini Coeff', 'Transparency Score']),
                                    x='Gini Coeff', y="Transparency Score", hover_data=["Country Name"], trendline='ols',
                                    trendline_color_override='red', hover_name="Country Name")
gini_gdp_corr = px.scatter(main_db[main_db["Year"] == 2018].dropna(subset=['Gini Coeff', 'GDP per capita']),
                           x='Gini Coeff', y="GDP per capita", hover_data=["Country Name"], trendline='ols',
                           trendline_color_override='red', hover_name="Country Name")
gini_distribution = px.histogram(mean_db.dropna(
    subset=["Region"]), x="Gini Coeff", nbins=10, color="Region")
gini_map = px.choropleth(mean_per_country(main_db).dropna(subset=[
                         'Region']), locations="Country Code", color="Gini Coeff", hover_name='Country Name')
per_income_group = px.scatter(incomegroup_db, x="Gini Coeff", y="Transparency Score",
                              color='IncomeGroup', size=np.log2(incomegroup_db['GDP per capita']), hover_name='IncomeGroup')
polar_plot = make_subplots(specs=[[{'type': 'polar'}]])
polar_plot.add_trace(go.Scatterpolar(
    r=region_db_2010_2019["Gini Coeff"], theta=region_db_2010_2019["Region"], name="Gini Coefficient"))
polar_plot.add_trace(go.Scatterpolar(
    r=region_db_2010_2019["GDP per capita"]/1000, theta=region_db_2010_2019["Region"], name="GDP per capita (k$)"))
# polar_plot.add_trace(go.Scatterpolar(
# r=region_db_2010_2019["Transparency Score"], theta=region_db_2010_2019["Region"], name="Transparency Score"))
violin_chart = px.violin(mean_db.dropna(
    subset=['Region']), y='Gini Coeff', color='Region', box=True)


# Tools for callbacks
all_countries = list(gini_meta['TableName'].unique())
all_countries.insert(0, 'All')

all_options = {"All": all_countries}
for region in gini_meta['Region'].unique():
    region_countries = sorted(list(
        gini_meta[gini_meta['Region'] == region]['TableName']))
    region_countries.insert(0, 'All')
    all_options[region] = region_countries

region_options = [{'label': k, 'value': k} for k in sorted(all_options.keys())]
country_options = [{'label': c, 'value': c} for c in all_countries]
indicators_list = ['Gini Coeff', 'Transparency Score',
                   'GDP per capita', 'Democracy Index']
indicator_options = [{'label': k, 'value': k} for k in indicators_list]
group_options = [{'label': 'Per Region', 'value': 'Region'},
                 {'label': 'Per Income Group', 'value': 'IncomeGroup'}, {'label': 'Regime Type', 'value': 'Regime Type'}]

app.layout = html.Div(style={'textAlign': 'center', 'width': '94%', 'padding-left': '3%'}, children=[
    html.H1(children='Inequality in the world', style={'textAlign': 'center'}),
    html.H6(children='This Dashboard is exploring inequality in the World and its evolution. The indicator used is the Gini Coefficient.\
         A low Gini coefficient means low level of inequality. For reference, the minimum in the data is 20.7 and the maximum is 65.8. \
             The Dashboard also explores the potential correlations between inequality and other indicators.'),
    'The references for the data is available at the bottom of the page.',
    dcc.Markdown('''
    ## Chart 1: Evolution of Gini Coefficient with time
    This Chart presents the mean Gini Coefficient average evolution per year. It can be filtered by Region and/or by Country.\n
    **WARNING**: data is not available for all countries every year so result on limited number of country may be difficult to interpret.
    '''),
    html.Br(),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='region-list',
                options=region_options,
                value='All'
            )
        ],
            style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='country-list',
                options=country_options
            )
        ],
            style={'width': '30%', 'float': 'right', 'display': 'inline-block', "padding-right": "10%"})
    ]),

    dcc.Graph(
        id='by-country',
        figure=gini_graph
    ),
    dcc.Markdown('''
    ## Chart 2: Evolution of Gini Coefficient per Country
    This Chart presents the Gini Coefficient average evolution per country per year. Several countries can be selected at the same time for comparison.
    '''),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='country-compare-list',
                options=country_options,
                value=['France'],
                multi=True
            )
        ], style={'width': '50%', 'display': 'inline-block'}),



        dcc.Graph(
            id='compare-country',
            figure=gini_graph
        ),
    ]),


    dcc.Markdown('''
    ## Chart 3: Gini Coefficient on a map
    Gini Coefficient per country on the map. The chart displays the average Gini coefficient on the selected period.
    '''),
    html.Div([
        dcc.Graph(
            id='gini-map',
            figure=gini_map
        ),
        html.Div([
            dcc.RangeSlider(
                id='years-slider',
                min=1967,
                max=2020,
                step=1,
                value=[1967, 2020],
                marks={i: str(i) for i in range(1970, 2021, 5)}
            )], style={"width": "80%", "padding-left": '10%'}
        )
    ]),
    html.Br(),
    html.Br(),
    html.Br(),
    dcc.Markdown('''
    ## Chart 4: Correlation between Gini Coefficient and other indicators
    The below 2 charts explore the correlation between Gini Coefficient and respectively Transparency Score and GDP per capita.
    Trendlines have been added to help visualise the correlation direction.
    '''),
    html.Div([
        dcc.Graph(
            id='gini-transparency-corr',
            figure=gini_transparency_corr
        )
    ], style={"height": "100%", "width": "45%", 'display': 'inline-block'}
    ),
    html.Div([
        dcc.Graph(
            id='gini-gdp-corr',
            figure=gini_gdp_corr
        )
    ], style={"height": "100%", "width": "45%", 'display': 'inline-block', 'float': 'right'}
    ),
    dcc.Markdown('''
    ## Chart 5: Average Gini Coefficient distribution
    This Chart presents the distribution of the Gini Coefficient average.
    It is split by region which allows to represent which regions have more countries with a low Gini Coefficient.
    '''),
    dcc.Graph(
        id='gini-distrib',
        figure=gini_distribution
    ),
    dcc.Markdown('''
    ## Chart 6: Comparison between Gini Coefficient and GDP
    This Chart presents Gini Coefficient average per region compared to the GDP per capita.
    The correlation saw in the scatter plot are not easy to catch here due to the mean per region.
    '''),
    dcc.Graph(
        id='polar-chart',
        figure=polar_plot
    ),
    dcc.Markdown('''
    ## Chart 7: Transparency relation to Gini Coefficient
    This Chart presents the relation between the Transparency score and the Gini Coefficient.
    The countries are grouped by Income Group and the scores are the average of available data.
    The size of the bubbles is the log2 of the GDP per capita average in the group.
    '''),
    dcc.Graph(
        id='bubble-chart',
        figure=per_income_group
    ),
    dcc.Markdown('''
    ## Chart 8: Different indicators distribution
    This Chart presents the distribution of an indicator split by different groups.\n
    The available indicators are *Gini Coefficient, Transparency Score, GDP per capita* and *Democracy Index*.  
    The available grouping are  *Region, Income Group* and Regime *Type*.
    '''),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='indicator-violin',
                options=indicator_options,
                value='Gini Coeff'
            )
        ],
            style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='group-violin',
                options=group_options,
                value='Region'
            )
        ],
            style={'width': '30%', 'float': 'right', 'display': 'inline-block', "padding-right": "10%"}),

        dcc.Graph(
            id='violin-chart',
            figure=violin_chart
        ),

    ]),
    html.Br(),
    html.Br(),
    dcc.Markdown(
        '''
    #### Data references:
    [Gini Coefficient from the World Bank] (https://data.worldbank.org/indicator/SI.POV.GINI)  
    [GDP per capita from the World Bank] (https://data.worldbank.org/indicator/NY.GDP.PCAP.CD)  
    [Democracy index from Wikipedia based on EIU] (https://en.wikipedia.org/wiki/Democracy_Index)  
    [Transparency score from Transparency International] (https://www.transparency.org/en/cpi/2020/index/nzl)  
    ''')


])

# Callbacks for "by-country" chart


@ app.callback(
    Output('country-list', 'options'),
    Input('region-list', 'value'))
def set_country_options(selected_region):
    return [{'label': i, 'value': i} for i in all_options[selected_region]]


@ app.callback(
    Output('country-list', 'value'),
    Input('country-list', 'options'))
def set_country_values(available_countries):
    return available_countries[0]['value']


@ app.callback(
    Output('by-country', 'figure'),
    Input('region-list', 'value'),
    Input('country-list', 'value'))
def set_display_children(selected_region, selected_country):
    if selected_region == 'All':
        out_df = mean_per_year(main_db)
    elif selected_country == 'All':
        country_list = all_options[selected_region].copy()
        country_list.remove('All')
        out_df = mean_per_year(
            main_db[main_db['Country Name'].isin(country_list)])
    else:
        out_df = mean_per_year(
            main_db[main_db['Country Name'] == selected_country])
    fig = px.scatter(out_df.dropna(
        subset=['Gini Coeff']), x="Year", y="Gini Coeff")

    return fig

# Callback for "compare-country" chart


@ app.callback(
    Output('compare-country', 'figure'),
    Input('country-compare-list', 'value'))
def set_display_children(list_country):
    if list_country:
        out_df = gini_db[gini_db["Country Name"].isin(list_country)].dropna()
        fig = px.line(out_df, x="Year", y='Gini Coeff',
                      color="Country Name")
        for d in fig.data:
            d.update(mode='markers+lines')
        return fig

# Callback for the Map


@ app.callback(
    Output('gini-map', 'figure'),
    Input('years-slider', 'value')


)
def update_map(years):
    start, end = years
    df_map = main_db[main_db['Year'] >= start]
    df_map = df_map[df_map['Year'] <= end]
    map_fig = px.choropleth(mean_per_country(df_map).dropna(subset=[
        'Region']), locations="Country Code", color="Gini Coeff", hover_name='Country Name')
    return map_fig
# Callback for "violin_chart":


@ app.callback(
    Output('violin-chart', 'figure'),
    Input('indicator-violin', 'value'),
    Input("group-violin", 'value')
)
def update_violin_chart(indicator, group):

    violin_fig = px.violin(mean_db.dropna(
        subset=[group]), y=indicator, color=group, box=True)

    return violin_fig


if __name__ == '__main__':
    app.run_server(debug=True)
