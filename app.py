import dash

from dash import dcc, html, Input, Output, State

import dash_bootstrap_components as dbc

import numpy as np

import pandas as pd

import plotly.graph_objs as go

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.preprocessing import StandardScaler

 

ASSET_POOLS = {

    "Commercial Loans": [

        "Small Business",

        "Middle Market",

        "Large Corporate",

        "Industry-Specific"

    ],

    "Consumer Loans": [

        "Mortgages",

        "Auto Loans",

        "Credit Cards",

        "Personal Loans",

        "Student Loans"

    ],

    "Commercial Real Estate": [

        "Office",

        "Retail",

        "Industrial",

        "Multifamily",

        "Hotel",

        "Healthcare"

    ],

    "Specialized Lending": [

        "Agricultural Loans",

        "Construction Loans",

        "Energy Sector",

        "Technology Sector",

        "Transportation"

    ]

}

 

ECONOMIC_FACTORS = [

    "GDP Growth Rate",

    "Unemployment Rate",

    "Inflation Rate"

]

 

DEFAULT_ECONOMIC_FACTORS = {

    "GDP Growth Rate": 2.5,

    "Unemployment Rate": 5.0,

    "Inflation Rate": 2.0

}

 

INITIAL_WEIGHTS = {

    "Commercial Loans": {

        "Small Business": [-0.4, 0.4, 0.2],

        "Middle Market": [-0.5, 0.3, 0.2],

        "Large Corporate": [-0.6, 0.2, 0.2],

        "Industry-Specific": [-0.5, 0.3, 0.2]

    },

    "Consumer Loans": {

        "Mortgages": [-0.3, 0.5, 0.2],

        "Auto Loans": [-0.3, 0.4, 0.3],

        "Credit Cards": [-0.2, 0.5, 0.3],

        "Personal Loans": [-0.3, 0.5, 0.2],

        "Student Loans": [-0.2, 0.6, 0.2]

    },

    "Commercial Real Estate": {

        "Office": [-0.5, 0.3, 0.2],

        "Retail": [-0.4, 0.4, 0.2],

        "Industrial": [-0.5, 0.3, 0.2],

        "Multifamily": [-0.3, 0.5, 0.2],

        "Hotel": [-0.4, 0.4, 0.2],

        "Healthcare": [-0.3, 0.3, 0.4]

    },

    "Specialized Lending": {

        "Agricultural Loans": [-0.4, 0.2, 0.4],

        "Construction Loans": [-0.5, 0.3, 0.2],

        "Energy Sector": [-0.5, 0.2, 0.3],

        "Technology Sector": [-0.6, 0.2, 0.2],

        "Transportation": [-0.4, 0.3, 0.3]

    }

}

 

def calculate_cecl_reserve(balance, avg_life, hist_loss_rate, prepayment_rate, economic_factors, weights, model_type):

    balance = float(balance)

    avg_life = float(avg_life)

    hist_loss_rate = float(hist_loss_rate) / 100

    prepayment_rate = float(prepayment_rate) / 100

    economic_factors = [float(ef) / 100 for ef in economic_factors]

 

    # Calculate the economic impact factor

    economic_impact = sum(factor * weight for factor, weight in zip(economic_factors, weights))

   

    # Adjust the historical loss rate based on economic factors

    adjusted_loss_rate = hist_loss_rate * (1 + economic_impact)

   

    X = np.array([adjusted_loss_rate, sum(economic_factors)]).reshape(1, -1)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

   

    base_loss = balance * adjusted_loss_rate * avg_life

 

    if model_type == "Linear Regression":

        model = LinearRegression()

        y = np.array([base_loss])

        model.fit(X_scaled, y)

        adjusted_loss = model.predict(X_scaled)[0]

    else:  # Logistic Regression

        model = LogisticRegression()

        y = np.array([0, 1])

        X_dummy = np.vstack([X_scaled, X_scaled])

        model.fit(X_dummy, y)

        prob = model.predict_proba(X_scaled)[0][1]

        adjusted_loss = base_loss * prob

 

    # Apply prepayment rate

    cecl_reserve = adjusted_loss * (1 - prepayment_rate)

   

    return max(cecl_reserve, 0)

 

def create_weight_heatmap(weights, title):

    df = pd.DataFrame(weights).T

    df.columns = ECONOMIC_FACTORS

   

    heatmap = go.Figure(data=go.Heatmap(

        z=df.values,

        x=df.columns,

        y=df.index,

        colorscale='RdBu',

        showscale=True,

        zmid=0

    ))

    heatmap.update_layout(

        title=title,

        xaxis_title='Economic Factor',

        yaxis_title='Asset Subcategory',

        height=400,

        margin=dict(l=50, r=50, t=50, b=50)

    )

    return heatmap

 

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

 

def info_icon(tooltip_text):

    return html.Span([

        " ",

        html.I(className="fas fa-info-circle", id=f"tooltip-{tooltip_text.replace(' ', '-')}"),

        dbc.Tooltip(tooltip_text, target=f"tooltip-{tooltip_text.replace(' ', '-')}")

    ])

 

app.layout = dbc.Container([

    html.H1("CECL Model Dashboard", className="my-4 text-center"),

   

    dbc.Row([

        dbc.Col([

            dbc.Card([

                dbc.CardHeader(html.H4("Model Parameters", className="mb-0")),

                dbc.CardBody([

                    html.Div([

                        dbc.Label("Model Type"),

                        dcc.Dropdown(

                            id="model-type",

                            options=[

                                {"label": "Linear Regression", "value": "Linear Regression"},

                                {"label": "Logistic Regression", "value": "Logistic Regression"}

                            ],

                            value="Linear Regression",

                        ),

                        info_icon("Choose between Linear Regression and Logistic Regression for CECL calculations. Each model uses different approaches to estimate expected credit losses.")

                    ], className="mb-3"),

                ])

            ], className="mb-3"),

           

            dbc.Card([

                dbc.CardHeader(html.H4("Economic Factors", className="mb-0")),

                dbc.CardBody([

                    html.Div([

                        dbc.Label(f"{factor} (%)"),

                        dbc.Input(id=f"ef-{factor.lower().replace(' ', '-')}", type="number", value=DEFAULT_ECONOMIC_FACTORS[factor], step=0.1),

                        info_icon(f"Current or forecasted {factor}. This affects the CECL calculations based on the economic factor weights.")

                    ], className="mb-3") for factor in ECONOMIC_FACTORS

                ])

            ], className="mb-3"),

           

            dbc.Button("Calculate Reserves", id="calculate-btn", color="primary", className="w-100")

        ], md=4),

       

        dbc.Col([

            dbc.Tabs([

                dbc.Tab(label="Asset Pools", children=[

                    html.Div([

                        html.H5(category, className="mt-3"),

                        dbc.Table([

                            html.Thead(html.Tr([html.Th("Subcategory"), html.Th("Balance ($)"), html.Th("Avg Life (years)"), html.Th("Historical Loss Rate (%)"), html.Th("Prepayment Rate (%)"), html.Th("CECL Reserve ($)")])),

                            html.Tbody([

                                html.Tr([

                                    html.Td(subcategory),

                                    html.Td(dbc.Input(id=f"balance-{category.lower().replace(' ', '-')}-{subcategory.lower().replace(' ', '-')}", type="number", value=1000000, step=1000)),

                                    html.Td(dbc.Input(id=f"avg-life-{category.lower().replace(' ', '-')}-{subcategory.lower().replace(' ', '-')}", type="number", value=5, step=0.1)),

                                    html.Td(dbc.Input(id=f"hist-loss-rate-{category.lower().replace(' ', '-')}-{subcategory.lower().replace(' ', '-')}", type="number", value=1, step=0.01)),

                                    html.Td(dbc.Input(id=f"prepayment-rate-{category.lower().replace(' ', '-')}-{subcategory.lower().replace(' ', '-')}", type="number", value=2, step=0.1)),

                                    html.Td(html.Div(id=f"cecl-reserve-{category.lower().replace(' ', '-')}-{subcategory.lower().replace(' ', '-')}"))

                                ]) for subcategory in subcategories

                            ])

                        ], bordered=True, hover=True, responsive=True, striped=True)

                    ]) for category, subcategories in ASSET_POOLS.items()

                ]),

                dbc.Tab(label="Economic Factor Weights", children=[

                    html.P("The heatmaps below show the impact of each economic factor on different asset categories. Blue indicates a negative relationship (factor increase leads to reserve decrease), while red indicates a positive relationship."),

                    *[html.Div([

                        html.H6(f"Economic Factor Weights for {category}", className="mt-3"),

                        dcc.Graph(id=f"weight-heatmap-{category.lower().replace(' ', '-')}")

                    ]) for category in ASSET_POOLS.keys()]

                ]),

                dbc.Tab(label="Results", children=[

                    html.Div(id="summary-stats", className="mt-4"),

                    dcc.Graph(id="reserve-chart"),

                ]),

            ])

        ], md=8)

    ]),

], fluid=True)

 

@app.callback(

    [Output(f"cecl-reserve-{category.lower().replace(' ', '-')}-{subcategory.lower().replace(' ', '-')}", "children")

     for category, subcategories in ASSET_POOLS.items() for subcategory in subcategories] +

    [Output("summary-stats", "children"), Output("reserve-chart", "figure")] +

    [Output(f"weight-heatmap-{category.lower().replace(' ', '-')}", "figure") for category in ASSET_POOLS.keys()],

    Input("calculate-btn", "n_clicks"),

    [State("model-type", "value")] +

    [State(f"ef-{factor.lower().replace(' ', '-')}", "value") for factor in ECONOMIC_FACTORS] +

    [State(f"balance-{category.lower().replace(' ', '-')}-{subcategory.lower().replace(' ', '-')}", "value")

     for category, subcategories in ASSET_POOLS.items() for subcategory in subcategories] +

    [State(f"avg-life-{category.lower().replace(' ', '-')}-{subcategory.lower().replace(' ', '-')}", "value")

     for category, subcategories in ASSET_POOLS.items() for subcategory in subcategories] +

    [State(f"hist-loss-rate-{category.lower().replace(' ', '-')}-{subcategory.lower().replace(' ', '-')}", "value")

     for category, subcategories in ASSET_POOLS.items() for subcategory in subcategories] +

    [State(f"prepayment-rate-{category.lower().replace(' ', '-')}-{subcategory.lower().replace(' ', '-')}", "value")

     for category, subcategories in ASSET_POOLS.items() for subcategory in subcategories]

)

def update_reserves(n_clicks, model_type, *args):

    if n_clicks is None:

        return [""] * sum(len(subcategories) for subcategories in ASSET_POOLS.values()) + [None, {}] + [go.Figure() for _ in ASSET_POOLS]

   

    economic_factors = args[:3]

    num_subcategories = sum(len(subcategories) for subcategories in ASSET_POOLS.values())

    balances = args[3:3+num_subcategories]

    avg_lives = args[3+num_subcategories:3+2*num_subcategories]

    hist_loss_rates = args[3+2*num_subcategories:3+3*num_subcategories]

    prepayment_rates = args[3+3*num_subcategories:3+4*num_subcategories]

 

    reserves = []

    total_balance = 0

    total_reserve = 0

   

    subcategory_index = 0

    for category, subcategories in ASSET_POOLS.items():

        for subcategory in subcategories:

            balance = balances[subcategory_index]

            avg_life = avg_lives[subcategory_index]

            hist_loss_rate = hist_loss_rates[subcategory_index]

            prepayment_rate = prepayment_rates[subcategory_index]

           

            if all(v is not None for v in [balance, avg_life, hist_loss_rate, prepayment_rate]):

                weights = INITIAL_WEIGHTS[category][subcategory]

                cecl_reserve = calculate_cecl_reserve(balance, avg_life, hist_loss_rate, prepayment_rate, economic_factors, weights, model_type)

                reserves.append(f"${int(cecl_reserve):,}")

                total_balance += balance

                total_reserve += cecl_reserve

            else:

                reserves.append("")

           

            subcategory_index += 1

 

    overall_reserve_rate = (total_reserve / total_balance) * 100 if total_balance > 0 else 0

   

    summary_stats = [

        html.H5("Summary Statistics"),

        html.P(f"Total Balance: ${int(total_balance):,}"),

        html.P(f"Total CECL Reserve: ${int(total_reserve):,}"),

        html.P(f"Overall Reserve Rate: {overall_reserve_rate:.2f}%"),

    ]

   

    fig = go.Figure(data=[go.Bar(

        x=[f"{category} - {subcategory}" for category, subcategories in ASSET_POOLS.items() for subcategory in subcategories],

        y=[int(float(reserve.replace('$', '').replace(',', ''))) if reserve else 0 for reserve in reserves],

        text=[f"${int(float(reserve.replace('$', '').replace(',', ''))):,}" if reserve else "$0" for reserve in reserves],

        textposition='auto',

    )])

    fig.update_layout(

        title="CECL Reserve by Asset Pool",

        xaxis_title="Asset Pool",

        yaxis_title="Reserve Amount ($)",

        xaxis={'categoryorder':'total descending'}

    )

   

    # Create heatmaps for each category

    category_heatmaps = [create_weight_heatmap(INITIAL_WEIGHTS[category], f"Economic Factor Weights for {category}") for category in ASSET_POOLS.keys()]

   

    return reserves + [summary_stats, fig] + category_heatmaps

 

if __name__ == "__main__":

    app.run_server(debug=True)