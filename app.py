import dash
from dash import dcc, html, Input, Output, State, callback, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objs as go

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Define loan pools and economic scenarios
COMMERCIAL_POOLS = {
    "C1": "CRE - Office", "C2": "CRE - Retail", "C3": "CRE - Industrial",
    "C4": "C&I - Large", "C5": "C&I - Middle", "C6": "Small Business", "C7": "Agricultural"
}
CONSUMER_POOLS = {
    "P1": "Res. Mortgages - Fixed", "P2": "Res. Mortgages - Adj.", "P3": "HELOCs",
    "P4": "Auto - New", "P5": "Auto - Used", "P6": "Credit Cards",
    "P7": "Personal - Secured", "P8": "Personal - Unsecured", "P9": "Student Loans"
}
ALL_POOLS = {**COMMERCIAL_POOLS, **CONSUMER_POOLS}
ECONOMIC_SCENARIOS = ["Baseline", "Adverse", "Severely Adverse"]

# Default data
DEFAULT_POOL_DATA = {
    pool_id: {
        "balance": 100_000_000, "default-prob": 0.02, "lgd": 0.35,
        "average-life": 5, "discount-rate": 0.05, "undrawn-percentage": 0.1
    } for pool_id in ALL_POOLS
}
DEFAULT_ECONOMIC_DATA = {
    "Baseline": {"gdp-growth": 0.02, "unemployment-rate": 5, "interest-rate": 0.03, "housing-price-index": 200},
    "Adverse": {"gdp-growth": -0.01, "unemployment-rate": 8, "interest-rate": 0.05, "housing-price-index": 180},
    "Severely Adverse": {"gdp-growth": -0.04, "unemployment-rate": 12, "interest-rate": 0.07, "housing-price-index": 160},
}

class CECLEngine:
    def __init__(self):
        self.economic_factors = pd.DataFrame(DEFAULT_ECONOMIC_DATA).T
        self.asset_pools = DEFAULT_POOL_DATA.copy()
        self.scenario_weights = {"Baseline": 0.4, "Adverse": 0.3, "Severely Adverse": 0.3}
        self.economic_sensitivities = {pool_id: {"gdp-growth": 0.7, "unemployment-rate": 0.5, "interest-rate": 0.8, "housing-price-index": 0.9} for pool_id in ALL_POOLS}

    def calculate_expected_loss(self, pool_id, scenario):
        pool_data = self.asset_pools[pool_id]
        economic_data = self.economic_factors.loc[scenario]
        economic_impact = sum(self.economic_sensitivities[pool_id][factor] * economic_data[factor] for factor in economic_data.index)
        pd_adjusted = min(1, max(0, pool_data['default-prob'] * (1 + economic_impact * 0.1)))
        lgd_adjusted = min(1, max(0, pool_data['lgd'] * (1 + economic_impact * 0.05)))
        ead = pool_data['balance'] * (1 + pool_data['undrawn-percentage'])
        return pd_adjusted * lgd_adjusted * ead

    def calculate_lifetime_ecl(self, pool_id):
        lifetime = self.asset_pools[pool_id]['average-life']
        return sum(
            self.scenario_weights[scenario] * sum(
                self.calculate_expected_loss(pool_id, scenario) / (1 + self.asset_pools[pool_id]['discount-rate'])**(year+1)
                for year in range(int(lifetime))
            ) for scenario in ECONOMIC_SCENARIOS
        )

calc_engine = CECLEngine()

def create_input_group(pool_id, pool_name):
    return dbc.Row([
        dbc.Col(html.Div(pool_name), width=2),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-balance"}, type="text", value=f"{DEFAULT_POOL_DATA[pool_id]['balance']:,}"), width=2),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-default-prob"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['default-prob']), width=1),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-lgd"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['lgd']), width=1),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-average-life"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['average-life']), width=2),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-discount-rate"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['discount-rate']), width=2),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-undrawn-percentage"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['undrawn-percentage']), width=2),
    ], className="mb-2")

def create_economic_inputs(scenario):
    return dbc.Row([
        dbc.Col(html.Div(scenario), width=2),
        dbc.Col(dbc.Input(id={"type": "economic-input", "id": f"{scenario}-gdp-growth"}, type="number", value=DEFAULT_ECONOMIC_DATA[scenario]['gdp-growth']), width=2),
        dbc.Col(dbc.Input(id={"type": "economic-input", "id": f"{scenario}-unemployment-rate"}, type="number", value=DEFAULT_ECONOMIC_DATA[scenario]['unemployment-rate']), width=2),
        dbc.Col(dbc.Input(id={"type": "economic-input", "id": f"{scenario}-interest-rate"}, type="number", value=DEFAULT_ECONOMIC_DATA[scenario]['interest-rate']), width=2),
        dbc.Col(dbc.Input(id={"type": "economic-input", "id": f"{scenario}-housing-price-index"}, type="number", value=DEFAULT_ECONOMIC_DATA[scenario]['housing-price-index']), width=2),
    ], className="mb-2")

app.layout = dbc.Container([
    html.H1("CECL Model Dashboard", className="text-center my-4"),
    html.Div([
        html.H4("Loan Pools"),
        dbc.Row([
            dbc.Col(html.Div("Pool Name"), width=2),
            dbc.Col(html.Div("Balance"), width=2),
            dbc.Col(html.Div("PD"), width=1),
            dbc.Col(html.Div("LGD"), width=1),
            dbc.Col(html.Div("Avg Life"), width=2),
            dbc.Col(html.Div("Discount Rate"), width=2),
            dbc.Col(html.Div("Undrawn %"), width=2),
        ], className="mb-2 font-weight-bold"),
        html.Div([create_input_group(pool_id, pool_name) for pool_id, pool_name in COMMERCIAL_POOLS.items()]),
        html.Div([create_input_group(pool_id, pool_name) for pool_id, pool_name in CONSUMER_POOLS.items()]),
        html.H4("Economic Scenarios", className="mt-4"),
        dbc.Row([
            dbc.Col(html.Div("Scenario"), width=2),
            dbc.Col(html.Div("GDP Growth"), width=2),
            dbc.Col(html.Div("Unemployment"), width=2),
            dbc.Col(html.Div("Interest Rate"), width=2),
            dbc.Col(html.Div("HPI"), width=2),
        ], className="mb-2 font-weight-bold"),
        html.Div([create_economic_inputs(scenario) for scenario in ECONOMIC_SCENARIOS]),
    ]),
    dbc.Button("Calculate", id="calculate-button", color="primary", className="mt-3"),
    html.Div(id="results-content", className="mt-4"),
])

@app.callback(
    Output("results-content", "children"),
    Input("calculate-button", "n_clicks"),
    State({"type": "pool-input", "id": ALL}, "value"),
    State({"type": "economic-input", "id": ALL}, "value"),
)
def update_results(n_clicks, pool_inputs, economic_inputs):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Update asset pools
    for i, pool_id in enumerate(ALL_POOLS):
        try:
            calc_engine.asset_pools[pool_id] = {
                'balance': float(pool_inputs[i*6].replace(',', '')),
                'default-prob': float(pool_inputs[i*6 + 1]),
                'lgd': float(pool_inputs[i*6 + 2]),
                'average-life': float(pool_inputs[i*6 + 3]),
                'discount-rate': float(pool_inputs[i*6 + 4]),
                'undrawn-percentage': float(pool_inputs[i*6 + 5]),
            }
        except (IndexError, ValueError):
            pass

    # Update economic factors
    if economic_inputs:
        for i, scenario in enumerate(ECONOMIC_SCENARIOS):
            try:
                calc_engine.economic_factors.loc[scenario] = {
                    'gdp-growth': float(economic_inputs[i*4]),
                    'unemployment-rate': float(economic_inputs[i*4 + 1]),
                    'interest-rate': float(economic_inputs[i*4 + 2]),
                    'housing-price-index': float(economic_inputs[i*4 + 3])
                }
            except (IndexError, ValueError):
                pass

    # Calculate ECL for each pool
    ecl_data = [(ALL_POOLS[pool_id], calc_engine.calculate_lifetime_ecl(pool_id)) for pool_id in ALL_POOLS]
    ecl_data.sort(key=lambda x: x[1], reverse=True)
    total_reserve = sum(ecl for _, ecl in ecl_data)

    # Generate charts and summary
    ecl_by_pool_chart = dcc.Graph(
        figure={
            'data': [go.Bar(x=[name for name, _ in ecl_data], y=[ecl for _, ecl in ecl_data])],
            'layout': go.Layout(title="Lifetime ECL by Pool", xaxis={'title': 'Pool'}, yaxis={'title': 'ECL ($)'})
        }
    )

    scenario_data = {scenario: [] for scenario in ECONOMIC_SCENARIOS}
    for pool_id in ALL_POOLS:
        for scenario in ECONOMIC_SCENARIOS:
            scenario_data[scenario].append(calc_engine.calculate_expected_loss(pool_id, scenario))

    ecl_by_scenario_chart = dcc.Graph(
        figure={
            'data': [go.Bar(name=scenario, x=list(ALL_POOLS.values()), y=ecls) for scenario, ecls in scenario_data.items()],
            'layout': go.Layout(title="ECL by Scenario and Pool", xaxis={'title': 'Pool'}, yaxis={'title': 'ECL ($)'}, barmode='group')
        }
    )

    ecl_summary = html.Div([
        html.H4("ECL Summary"),
        html.Ul([html.Li(f"{name}: ${ecl:,.2f}") for name, ecl in ecl_data]),
        html.H4(f"Total Reserve: ${total_reserve:,.2f}", className="mt-3")
    ])

    return html.Div([ecl_by_pool_chart, ecl_by_scenario_chart, ecl_summary])

@app.callback(
    Output({"type": "pool-input", "id": ALL}, "value"),
    Input({"type": "pool-input", "id": ALL}, "value"),
    State({"type": "pool-input", "id": ALL}, "id"),
)
def format_number(values, ids):
    return [f"{float(value.replace(',', '')):,.0f}" if "-balance" in id_dict["id"] else value for value, id_dict in zip(values, ids)]

if __name__ == '__main__':
    app.run_server(debug=True)
