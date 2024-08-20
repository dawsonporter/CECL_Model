import dash
from dash import dcc, html, Input, Output, State, callback, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objs as go

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)
server = app.server

# Define loan pools and economic scenarios
COMMERCIAL_POOLS = {
    "C1": "CRE - Office",
    "C2": "CRE - Retail",
    "C3": "CRE - Industrial",
    "C4": "CRE - Multi-Family",
    "C5": "C&I - Large",
    "C6": "C&I - Middle",
    "C7": "Small Business",
    "C8": "Agricultural"
}
CONSUMER_POOLS = {
    "P1": "Residential Mortgages",
    "P2": "Auto Loans",
    "P3": "Credit Cards",
    "P4": "Personal Loans"
}
ALL_POOLS = {**COMMERCIAL_POOLS, **CONSUMER_POOLS}
ECONOMIC_SCENARIOS = ["Baseline", "Adverse", "Severely Adverse"]

# Default data with more realistic loan balances (in millions)
DEFAULT_POOL_DATA = {
    "C1": {"balance": 2000, "default-prob": 2, "lgd": 35, "average-life": 7, "discount-rate": 5, "undrawn-percentage": 10},
    "C2": {"balance": 1800, "default-prob": 2.5, "lgd": 40, "average-life": 6, "discount-rate": 5, "undrawn-percentage": 15},
    "C3": {"balance": 1500, "default-prob": 1.5, "lgd": 30, "average-life": 8, "discount-rate": 5, "undrawn-percentage": 5},
    "C4": {"balance": 2500, "default-prob": 1.8, "lgd": 25, "average-life": 10, "discount-rate": 5, "undrawn-percentage": 5},
    "C5": {"balance": 3000, "default-prob": 1, "lgd": 45, "average-life": 5, "discount-rate": 5, "undrawn-percentage": 20},
    "C6": {"balance": 2000, "default-prob": 3, "lgd": 50, "average-life": 4, "discount-rate": 5, "undrawn-percentage": 25},
    "C7": {"balance": 1000, "default-prob": 4, "lgd": 55, "average-life": 3, "discount-rate": 5, "undrawn-percentage": 30},
    "C8": {"balance": 500, "default-prob": 3.5, "lgd": 45, "average-life": 5, "discount-rate": 5, "undrawn-percentage": 10},
    "P1": {"balance": 5000, "default-prob": 0.5, "lgd": 20, "average-life": 15, "discount-rate": 4, "undrawn-percentage": 0},
    "P2": {"balance": 1500, "default-prob": 2, "lgd": 40, "average-life": 5, "discount-rate": 6, "undrawn-percentage": 0},
    "P3": {"balance": 800, "default-prob": 5, "lgd": 70, "average-life": 2, "discount-rate": 10, "undrawn-percentage": 60},
    "P4": {"balance": 500, "default-prob": 4, "lgd": 60, "average-life": 3, "discount-rate": 8, "undrawn-percentage": 0},
}

DEFAULT_ECONOMIC_DATA = {
    "Baseline": {"gdp-growth": 2, "unemployment-rate": 5, "interest-rate": 3, "housing-price-index": 200},
    "Adverse": {"gdp-growth": -1, "unemployment-rate": 8, "interest-rate": 5, "housing-price-index": 180},
    "Severely Adverse": {"gdp-growth": -4, "unemployment-rate": 12, "interest-rate": 7, "housing-price-index": 160},
}

# Economic sensitivities (impact multipliers)
ECONOMIC_SENSITIVITIES = {
    "Commercial": {"gdp-growth": 1.2, "unemployment-rate": 0.8, "interest-rate": 1.5, "housing-price-index": 0.5},
    "Consumer": {"gdp-growth": 0.8, "unemployment-rate": 1.2, "interest-rate": 1.0, "housing-price-index": 1.5}
}

class CECLEngine:
    def __init__(self):
        self.economic_factors = pd.DataFrame(DEFAULT_ECONOMIC_DATA).T
        self.asset_pools = DEFAULT_POOL_DATA.copy()
        self.scenario_weights = {"Baseline": 0.4, "Adverse": 0.3, "Severely Adverse": 0.3}

    def calculate_expected_loss(self, pool_id, scenario):
        pool_data = self.asset_pools[pool_id]
        economic_data = self.economic_factors.loc[scenario]
        pool_type = "Commercial" if pool_id.startswith("C") else "Consumer"
        
        economic_impact = sum(
            ECONOMIC_SENSITIVITIES[pool_type][factor] * economic_data[factor] / 100
            for factor in economic_data.index
        )
        
        pd_adjusted = min(100, max(0, pool_data['default-prob'] * (1 + economic_impact)))
        lgd_adjusted = min(100, max(0, pool_data['lgd'] * (1 + economic_impact * 0.5)))
        ead = pool_data['balance'] * (1 + pool_data['undrawn-percentage'] / 100)
        
        return (pd_adjusted / 100) * (lgd_adjusted / 100) * ead

    def calculate_lifetime_ecl(self, pool_id):
        lifetime = self.asset_pools[pool_id]['average-life']
        return sum(
            self.scenario_weights[scenario] * sum(
                self.calculate_expected_loss(pool_id, scenario) / (1 + self.asset_pools[pool_id]['discount-rate'] / 100)**(year+1)
                for year in range(int(lifetime))
            ) for scenario in ECONOMIC_SCENARIOS
        )

calc_engine = CECLEngine()

def create_input_group(pool_id, pool_name):
    return dbc.Row([
        dbc.Col(html.Div(pool_name, className="fw-bold"), width=2),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-balance"}, type="text", value=f"{DEFAULT_POOL_DATA[pool_id]['balance']:,}", className="form-control-sm"), width=2),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-default-prob"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['default-prob'], className="form-control-sm"), width=1),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-lgd"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['lgd'], className="form-control-sm"), width=1),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-average-life"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['average-life'], className="form-control-sm"), width=2),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-discount-rate"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['discount-rate'], className="form-control-sm"), width=2),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-undrawn-percentage"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['undrawn-percentage'], className="form-control-sm"), width=2),
    ], className="mb-2 align-items-center")

def create_economic_inputs(scenario):
    return dbc.Row([
        dbc.Col(html.Div(scenario, className="fw-bold"), width=2),
        dbc.Col(dbc.Input(id={"type": "economic-input", "id": f"{scenario}-gdp-growth"}, type="number", value=DEFAULT_ECONOMIC_DATA[scenario]['gdp-growth'], className="form-control-sm"), width=2),
        dbc.Col(dbc.Input(id={"type": "economic-input", "id": f"{scenario}-unemployment-rate"}, type="number", value=DEFAULT_ECONOMIC_DATA[scenario]['unemployment-rate'], className="form-control-sm"), width=2),
        dbc.Col(dbc.Input(id={"type": "economic-input", "id": f"{scenario}-interest-rate"}, type="number", value=DEFAULT_ECONOMIC_DATA[scenario]['interest-rate'], className="form-control-sm"), width=2),
        dbc.Col(dbc.Input(id={"type": "economic-input", "id": f"{scenario}-housing-price-index"}, type="number", value=DEFAULT_ECONOMIC_DATA[scenario]['housing-price-index'], className="form-control-sm"), width=2),
    ], className="mb-2 align-items-center")

app.layout = dbc.Container([
    html.H1("CECL Model Dashboard", className="text-center my-4"),
    dbc.Card([
        dbc.CardHeader(html.H4("Loan Pools", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Div("Pool Name", className="fw-bold"), width=2),
                dbc.Col(html.Div("Balance ($M)", className="fw-bold"), width=2),
                dbc.Col(html.Div("PD (%)", className="fw-bold"), width=1),
                dbc.Col(html.Div("LGD (%)", className="fw-bold"), width=1),
                dbc.Col(html.Div("Avg Life (Years)", className="fw-bold"), width=2),
                dbc.Col(html.Div("Discount Rate (%)", className="fw-bold"), width=2),
                dbc.Col(html.Div("Undrawn (%)", className="fw-bold"), width=2),
            ], className="mb-2"),
            html.Div([create_input_group(pool_id, pool_name) for pool_id, pool_name in COMMERCIAL_POOLS.items()]),
            html.Div([create_input_group(pool_id, pool_name) for pool_id, pool_name in CONSUMER_POOLS.items()]),
        ]),
    ], className="mb-4"),
    dbc.Card([
        dbc.CardHeader(html.H4("Economic Scenarios", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Div("Scenario", className="fw-bold"), width=2),
                dbc.Col(html.Div("GDP Growth (%)", className="fw-bold"), width=2),
                dbc.Col(html.Div("Unemployment (%)", className="fw-bold"), width=2),
                dbc.Col(html.Div("Interest Rate (%)", className="fw-bold"), width=2),
                dbc.Col(html.Div("HPI", className="fw-bold"), width=2),
            ], className="mb-2"),
            html.Div([create_economic_inputs(scenario) for scenario in ECONOMIC_SCENARIOS]),
        ]),
    ], className="mb-4"),
    dbc.Button("Calculate", id="calculate-button", color="primary", className="mb-4"),
    html.Div(id="results-content"),
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
    ecl_data = [(pool_id, ALL_POOLS[pool_id], calc_engine.calculate_lifetime_ecl(pool_id)) for pool_id in ALL_POOLS]
    ecl_data.sort(key=lambda x: x[2], reverse=True)
    total_reserve = sum(ecl for _, _, ecl in ecl_data)

    # Generate charts and summary
    ecl_by_pool_chart = dcc.Graph(
        figure={
            'data': [go.Bar(x=[name for _, name, _ in ecl_data], y=[ecl for _, _, ecl in ecl_data])],
            'layout': go.Layout(title="Lifetime ECL by Pool", xaxis={'title': 'Pool'}, yaxis={'title': 'ECL ($M)'})
        }
    )

    scenario_data = {scenario: [] for scenario in ECONOMIC_SCENARIOS}
    for pool_id in ALL_POOLS:
        for scenario in ECONOMIC_SCENARIOS:
            scenario_data[scenario].append(calc_engine.calculate_expected_loss(pool_id, scenario))

    ecl_by_scenario_chart = dcc.Graph(
        figure={
            'data': [go.Bar(name=scenario, x=list(ALL_POOLS.values()), y=ecls) for scenario, ecls in scenario_data.items()],
            'layout': go.Layout(title="ECL by Scenario and Pool", xaxis={'title': 'Pool'}, yaxis={'title': 'ECL ($M)'}, barmode='group')
        }
    )

    # Create a more detailed summary
    commercial_ecl = sum(ecl for pool_id, _, ecl in ecl_data if pool_id.startswith('C'))
    consumer_ecl = sum(ecl for pool_id, _, ecl in ecl_data if pool_id.startswith('P'))
    total_balance = sum(calc_engine.asset_pools[pool_id]['balance'] for pool_id in ALL_POOLS)
    
    summary_table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Category"),
                html.Th("Total Balance ($M)"),
                html.Th("ECL ($M)"),
                html.Th("ECL Coverage (%)")
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td("Commercial"),
                html.Td(f"{sum(calc_engine.asset_pools[pool_id]['balance'] for pool_id in COMMERCIAL_POOLS):,.2f}"),
                html.Td(f"{commercial_ecl:,.2f}"),
                html.Td(f"{commercial_ecl / sum(calc_engine.asset_pools[pool_id]['balance'] for pool_id in COMMERCIAL_POOLS) * 100:.2f}%")
            ]),
            html.Tr([
                html.Td("Consumer"),
                html.Td(f"{sum(calc_engine.asset_pools[pool_id]['balance'] for pool_id in CONSUMER_POOLS):,.2f}"),
                html.Td(f"{consumer_ecl:,.2f}"),
                html.Td(f"{consumer_ecl / sum(calc_engine.asset_pools[pool_id]['balance'] for pool_id in CONSUMER_POOLS) * 100:.2f}%")
            ]),
            html.Tr([
                html.Td("Total", className="fw-bold"),
                html.Td(f"{total_balance:,.2f}", className="fw-bold"),
                html.Td(f"{total_reserve:,.2f}", className="fw-bold"),
                html.Td(f"{total_reserve / total_balance * 100:.2f}%", className="fw-bold")
            ])
        ])
    ], bordered=True, hover=True, striped=True, className="mt-4")

    # Top 5 pools by ECL
    top_5_ecl = ecl_data[:5]
    top_5_table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Pool"),
                html.Th("ECL ($M)"),
                html.Th("% of Total ECL")
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(name),
                html.Td(f"{ecl:,.2f}"),
                html.Td(f"{ecl / total_reserve * 100:.2f}%")
            ]) for _, name, ecl in top_5_ecl
        ])
    ], bordered=True, hover=True, striped=True, className="mt-4")

    ecl_summary = dbc.Card([
        dbc.CardHeader(html.H4("ECL Summary", className="mb-0")),
        dbc.CardBody([
            summary_table,
            html.H5("Top 5 Pools by ECL", className="mt-4"),
            top_5_table
        ])
    ])

    return dbc.Row([
        dbc.Col(ecl_by_pool_chart, md=6),
        dbc.Col(ecl_by_scenario_chart, md=6),
        dbc.Col(ecl_summary, md=12, className="mt-4")
    ])

@app.callback(
    Output({"type": "pool-input", "id": ALL}, "value"),
    Input({"type": "pool-input", "id": ALL}, "value"),
    State({"type": "pool-input", "id": ALL}, "id"),
)
def format_number(values, ids):
    return [f"{float(value.replace(',', '')):,.0f}" if "-balance" in id_dict["id"] else value for value, id_dict in zip(values, ids)]

if __name__ == '__main__':
    app.run_server(debug=True)
