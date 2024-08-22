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
    "C1": "CRE - Office", "C2": "CRE - Retail", "C3": "CRE - Industrial", "C4": "CRE - Multi-Family",
    "C5": "C&I - Large", "C6": "C&I - Middle", "C7": "Small Business", "C8": "Agricultural"
}
CONSUMER_POOLS = {
    "P1": "Residential Mortgages", "P2": "Auto Loans", "P3": "Credit Cards", "P4": "Personal Loans"
}
ALL_POOLS = {**COMMERCIAL_POOLS, **CONSUMER_POOLS}
ECONOMIC_SCENARIOS = ["Baseline", "Adverse", "Severely Adverse"]

# Default data with more realistic loan balances (in millions) and added prepayment rates
DEFAULT_POOL_DATA = {
    "C1": {"balance": 2000, "default-prob": 2.0, "lgd": 35.0, "original-term": 10, "discount-rate": 5.0, "undrawn-percentage": 10, "prepayment-rate": 5},
    "C2": {"balance": 1800, "default-prob": 2.5, "lgd": 40.0, "original-term": 8, "discount-rate": 5.0, "undrawn-percentage": 15, "prepayment-rate": 4},
    "C3": {"balance": 1500, "default-prob": 1.5, "lgd": 30.0, "original-term": 12, "discount-rate": 5.0, "undrawn-percentage": 5, "prepayment-rate": 3},
    "C4": {"balance": 2500, "default-prob": 1.8, "lgd": 25.0, "original-term": 15, "discount-rate": 5.0, "undrawn-percentage": 5, "prepayment-rate": 6},
    "C5": {"balance": 3000, "default-prob": 1.0, "lgd": 45.0, "original-term": 7, "discount-rate": 5.0, "undrawn-percentage": 20, "prepayment-rate": 8},
    "C6": {"balance": 2000, "default-prob": 3.0, "lgd": 50.0, "original-term": 5, "discount-rate": 5.0, "undrawn-percentage": 25, "prepayment-rate": 7},
    "C7": {"balance": 1000, "default-prob": 4.0, "lgd": 55.0, "original-term": 4, "discount-rate": 5.0, "undrawn-percentage": 30, "prepayment-rate": 10},
    "C8": {"balance": 500, "default-prob": 3.5, "lgd": 45.0, "original-term": 6, "discount-rate": 5.0, "undrawn-percentage": 10, "prepayment-rate": 2},
    "P1": {"balance": 5000, "default-prob": 0.5, "lgd": 20.0, "original-term": 30, "discount-rate": 4.0, "undrawn-percentage": 0, "prepayment-rate": 12},
    "P2": {"balance": 1500, "default-prob": 2.0, "lgd": 40.0, "original-term": 6, "discount-rate": 6.0, "undrawn-percentage": 0, "prepayment-rate": 15},
    "P3": {"balance": 800, "default-prob": 5.0, "lgd": 70.0, "original-term": 3, "discount-rate": 10.0, "undrawn-percentage": 60, "prepayment-rate": 20},
    "P4": {"balance": 500, "default-prob": 4.0, "lgd": 60.0, "original-term": 4, "discount-rate": 8.0, "undrawn-percentage": 0, "prepayment-rate": 8},
}

DEFAULT_ECONOMIC_DATA = {
    "Baseline": {"gdp-growth": 2.0, "unemployment-rate": 5.0, "fed-funds-rate": 3.00, "housing-price-index": 200},
    "Adverse": {"gdp-growth": -1.0, "unemployment-rate": 8.0, "fed-funds-rate": 5.00, "housing-price-index": 180},
    "Severely Adverse": {"gdp-growth": -4.0, "unemployment-rate": 12.0, "fed-funds-rate": 7.00, "housing-price-index": 160},
}

# Economic sensitivities (impact multipliers)
ECONOMIC_SENSITIVITIES = {
    "Commercial": {"gdp-growth": 0.7, "unemployment-rate": 0.8, "fed-funds-rate": 1.5, "housing-price-index": 0.5},
    "Consumer": {"gdp-growth": 0.8, "unemployment-rate": 1.2, "fed-funds-rate": 1.0, "housing-price-index": 1.5}
}

class CECLEngine:
    def __init__(self):
        self.economic_factors = pd.DataFrame(DEFAULT_ECONOMIC_DATA).T
        self.asset_pools = DEFAULT_POOL_DATA.copy()
        self.scenario_weights = {"Baseline": 0.4, "Adverse": 0.3, "Severely Adverse": 0.3}
        self.pd_multiplier = 0.6
        self.lgd_multiplier = 0.6

    def calculate_expected_loss(self, pool_id, scenario, year):
        pool_data = self.asset_pools[pool_id]
        economic_data = self.economic_factors.loc[scenario]
        pool_type = "Commercial" if pool_id.startswith('C') else "Consumer"
        
        # Calculate economic impact
        economic_impact = 0
        for factor, sensitivity in ECONOMIC_SENSITIVITIES[pool_type].items():
            if factor in ['gdp-growth', 'housing-price-index']:
                # For GDP and HPI, higher values reduce risk
                impact = -sensitivity * (economic_data[factor] - DEFAULT_ECONOMIC_DATA['Baseline'][factor]) / 100
            else:
                # For unemployment and fed funds rate, higher values increase risk
                impact = sensitivity * (economic_data[factor] - DEFAULT_ECONOMIC_DATA['Baseline'][factor]) / 100
            economic_impact += impact

        # Adjust PD and LGD based on economic impact and multipliers
        pd_adjusted = min(1, max(0, pool_data['default-prob'] / 100 * (1 + economic_impact) * self.pd_multiplier))
        lgd_adjusted = min(1, max(0, pool_data['lgd'] / 100 * (1 + economic_impact * 0.5) * self.lgd_multiplier))
        
        # Calculate remaining balance considering prepayments
        remaining_balance = pool_data['balance'] * (1 - pool_data['prepayment-rate'] / 100) ** year
        ead = remaining_balance * (1 + pool_data['undrawn-percentage'] / 100)
        
        return pd_adjusted * lgd_adjusted * ead

    def calculate_lifetime_ecl(self, pool_id):
        pool_data = self.asset_pools[pool_id]
        original_term = int(pool_data['original-term'])
        prepayment_rate = pool_data['prepayment-rate'] / 100
        discount_rate = pool_data['discount-rate'] / 100

        total_ecl = 0
        for year in range(original_term):
            year_ecl = sum(
                self.scenario_weights[scenario] * self.calculate_expected_loss(pool_id, scenario, year)
                for scenario in ECONOMIC_SCENARIOS
            )
            
            total_ecl += year_ecl / (1 + discount_rate) ** (year + 1)

            # Break if remaining balance is less than 1% of original balance
            if pool_data['balance'] * (1 - prepayment_rate) ** (year + 1) < 0.01 * pool_data['balance']:
                break

        return total_ecl

calc_engine = CECLEngine()

def create_input_group(pool_id, pool_name):
    return dbc.Row([
        dbc.Col(html.Div(pool_name, className="fw-bold text-center"), width=12, md=3, className="px-1"),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-balance"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['balance'], className="form-control text-center"), width=6, md=1, className="px-1"),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-default-prob"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['default-prob'], step=0.1, className="form-control text-center"), width=6, md=1, className="px-1"),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-lgd"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['lgd'], step=0.1, className="form-control text-center"), width=6, md=1, className="px-1"),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-original-term"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['original-term'], className="form-control text-center"), width=6, md=1, className="px-1"),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-discount-rate"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['discount-rate'], step=0.1, className="form-control text-center"), width=6, md=1, className="px-1"),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-undrawn-percentage"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['undrawn-percentage'], className="form-control text-center"), width=6, md=1, className="px-1"),
        dbc.Col(dbc.Input(id={"type": "pool-input", "id": f"{pool_id}-prepayment-rate"}, type="number", value=DEFAULT_POOL_DATA[pool_id]['prepayment-rate'], className="form-control text-center"), width=6, md=1, className="px-1"),
    ], className="mb-1 align-items-center g-0")

def create_economic_inputs(scenario):
    return dbc.Row([
        dbc.Col(html.Div(scenario, className="fw-bold text-center"), width=12, md=3, className="px-1"),
        dbc.Col(dbc.Input(id={"type": "economic-input", "id": f"{scenario}-gdp-growth"}, type="number", value=DEFAULT_ECONOMIC_DATA[scenario]['gdp-growth'], step=0.1, className="form-control text-center"), width=6, md=2, className="px-1"),
        dbc.Col(dbc.Input(id={"type": "economic-input", "id": f"{scenario}-unemployment-rate"}, type="number", value=DEFAULT_ECONOMIC_DATA[scenario]['unemployment-rate'], step=0.1, className="form-control text-center"), width=6, md=2, className="px-1"),
        dbc.Col(dbc.Input(id={"type": "economic-input", "id": f"{scenario}-fed-funds-rate"}, type="number", value=DEFAULT_ECONOMIC_DATA[scenario]['fed-funds-rate'], step=0.01, className="form-control text-center"), width=6, md=2, className="px-1"),
        dbc.Col(dbc.Input(id={"type": "economic-input", "id": f"{scenario}-housing-price-index"}, type="number", value=DEFAULT_ECONOMIC_DATA[scenario]['housing-price-index'], className="form-control text-center"), width=6, md=2, className="px-1"),
    ], className="mb-1 align-items-center g-0")

def create_weights_and_multipliers_inputs():
    return dbc.Card([
        dbc.CardHeader(html.H5("Weights and Multipliers", className="mb-0 text-center")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div("PD Multiplier", className="fw-bold text-center"),
                    dbc.Input(id="pd-multiplier", type="number", value=0.6, min=0, max=2, step=0.1, className="form-control text-center")
                ], width=12, md=2, className="px-1"),
                dbc.Col([
                    html.Div("LGD Multiplier", className="fw-bold text-center"),
                    dbc.Input(id="lgd-multiplier", type="number", value=0.6, min=0, max=2, step=0.1, className="form-control text-center")
                ], width=12, md=2, className="px-1"),
                dbc.Col([
                    html.Div("Baseline Weight", className="fw-bold text-center"),
                    dbc.Input(id="baseline-weight", type="number", value=0.4, min=0, max=1, step=0.1, className="form-control text-center")
                ], width=12, md=2, className="px-1"),
                dbc.Col([
                    html.Div("Adverse Weight", className="fw-bold text-center"),
                    dbc.Input(id="adverse-weight", type="number", value=0.3, min=0, max=1, step=0.1, className="form-control text-center")
                ], width=12, md=2, className="px-1"),
                dbc.Col([
                    html.Div("Severely Adverse Weight", className="fw-bold text-center"),
                    dbc.Input(id="severely-adverse-weight", type="number", value=0.3, min=0, max=1, step=0.1, className="form-control text-center")
                ], width=12, md=2, className="px-1"),
            ], className="g-0 align-items-end"),
        ], className="py-2"),
    ], className="mb-3")

def create_model_explanation():
    return html.Div([
        html.H2("CECL Model Explanation", className="mb-4 text-center"),
        html.H3("Overview", className="mb-3"),
        html.P("The Current Expected Credit Loss (CECL) model is an accounting standard that requires an estimate of expected credit losses to be made at the time a financial instrument is first recognized. This model calculates the lifetime expected credit losses for various loan pools under different economic scenarios."),
        html.H3("Input Parameters", className="mb-3"),
        html.H4("Loan Pool Inputs:", className="mb-2"),
        html.Ul([
            html.Li("Balance: The total outstanding balance of the loan pool (in millions of dollars)."),
            html.Li("Default Probability (PD): The likelihood of default over the life of the loan, expressed as a percentage."),
            html.Li("Loss Given Default (LGD): The portion of the loan balance that is expected to be lost if a default occurs, expressed as a percentage."),
            html.Li("Original Term: The original term of the loans in the pool, in years."),
            html.Li("Discount Rate: The rate used to discount future cash flows, expressed as a percentage."),
            html.Li("Undrawn Percentage: The portion of committed but undrawn balances, expressed as a percentage of the total commitment."),
            html.Li("Prepayment Rate: The expected rate of early repayment, expressed as a percentage of the remaining balance per year.")
        ]),
        html.H4("Economic Scenario Inputs:", className="mb-2"),
        html.Ul([
            html.Li("GDP Growth: The annual growth rate of Gross Domestic Product, expressed as a percentage."),
            html.Li("Unemployment Rate: The percentage of the labor force that is unemployed."),
            html.Li("Fed Funds Rate: The target interest rate set by the Federal Reserve, expressed as a percentage."),
            html.Li("Housing Price Index: An index representing the overall level of housing prices.")
        ]),
        html.H4("Weights and Multipliers:", className="mb-2"),
        html.Ul([
            html.Li("PD Multiplier: A factor to adjust the calculated Probability of Default across all pools. Default value is 0.6, providing a more conservative estimate."),
            html.Li("LGD Multiplier: A factor to adjust the calculated Loss Given Default across all pools. Default value is 0.6, providing a more conservative estimate."),
            html.Li("Economic Scenario Weights: The relative importance given to each economic scenario (Baseline, Adverse, Severely Adverse) in the final ECL calculation.")
        ]),
        html.H3("Calculation Methodology", className="mb-3"),
        html.P("The CECL calculation involves the following steps:"),
        html.Ol([
            html.Li("Adjust default probabilities and loss given default based on economic scenarios and multipliers."),
            html.Li("Calculate expected loss for each year of the loan's life under each economic scenario."),
            html.Li("Apply the prepayment rate to reduce the outstanding balance each year."),
            html.Li("Discount the expected losses to present value."),
            html.Li("Weight the losses from different economic scenarios using the specified weights."),
            html.Li("Sum the weighted, discounted expected losses to arrive at the lifetime ECL.")
        ]),
        html.H3("Impact of Economic Factors", className="mb-3"),
        html.P("The model considers the following economic impacts:"),
        html.Ul([
            html.Li("GDP Growth: Higher GDP growth reduces risk, leading to lower ECL."),
            html.Li("Unemployment Rate: Higher unemployment increases risk, leading to higher ECL."),
            html.Li("Fed Funds Rate: Higher rates generally increase risk, leading to higher ECL."),
            html.Li("Housing Price Index (HPI): Higher HPI reduces risk, leading to lower ECL.")
        ]),
        html.H3("PD and LGD Multipliers", className="mb-3"),
        html.P("The PD and LGD multipliers allow for additional adjustment of the calculated Probability of Default and Loss Given Default values. The default value of 0.6 for both multipliers provides a more conservative estimate, reducing the PD and LGD by 40%. This can be used to stress test the model or to account for additional factors not directly captured by the economic scenarios."),
        html.H3("Economic Scenario Weighting", className="mb-3"),
        html.P("The model uses user-defined weights for economic scenarios. These weights determine the relative importance of each scenario in the final ECL calculation. The sum of weights should equal 1."),
        html.H3("Economic Sensitivities", className="mb-3"),
        html.P("Different loan pools have different sensitivities to economic factors. The model uses the following sensitivity multipliers:"),
        html.Div([
            dbc.Table([
                html.Thead([
                    html.Tr([html.Th("Factor", className="text-center")] + [html.Th(pool_type, className="text-center") for pool_type in ECONOMIC_SENSITIVITIES.keys()])
                ]),
                html.Tbody([
                    html.Tr([html.Td(factor, className="text-center")] + [html.Td(f"{ECONOMIC_SENSITIVITIES[pool_type][factor]:.2f}", className="text-center") for pool_type in ECONOMIC_SENSITIVITIES.keys()])
                    for factor in ECONOMIC_SENSITIVITIES["Commercial"].keys()
                ])
            ], bordered=True, hover=True, striped=True, className="mt-3")
        ], className="d-flex justify-content-center"),
        html.H3("Typical Input Ranges", className="mb-3"),
        html.P("The following table provides typical ranges for input parameters across different loan pools:"),
        html.Div([
            dbc.Table([
                html.Thead([
                    html.Tr([html.Th(col, className="text-center") for col in ["Pool Type", "PD Range (%)", "LGD Range (%)", "Average Life (Years)", "Discount Rate (%)", "Undrawn (%)", "Prepayment (%)"]])
                ]),
                html.Tbody([
                    html.Tr([html.Td("CRE", className="text-center"), html.Td("1.0 - 3.0", className="text-center"), html.Td("20 - 45", className="text-center"), html.Td("5 - 10", className="text-center"), html.Td("4 - 6", className="text-center"), html.Td("5 - 20", className="text-center"), html.Td("2 - 10", className="text-center")]),
                    html.Tr([html.Td("C&I", className="text-center"), html.Td("0.5 - 4.0", className="text-center"), html.Td("30 - 60", className="text-center"), html.Td("3 - 7", className="text-center"), html.Td("4 - 7", className="text-center"), html.Td("20 - 40", className="text-center"), html.Td("5 - 15", className="text-center")]),
                    html.Tr([html.Td("Small Business", className="text-center"), html.Td("2.0 - 6.0", className="text-center"), html.Td("40 - 70", className="text-center"), html.Td("2 - 5", className="text-center"), html.Td("5 - 8", className="text-center"), html.Td("10 - 30", className="text-center"), html.Td("5 - 20", className="text-center")]),
                    html.Tr([html.Td("Residential Mortgages", className="text-center"), html.Td("0.2 - 1.0", className="text-center"), html.Td("10 - 30", className="text-center"), html.Td("15 - 30", className="text-center"), html.Td("3 - 5", className="text-center"), html.Td("0 - 5", className="text-center"), html.Td("5 - 15", className="text-center")]),
                    html.Tr([html.Td("Auto Loans", className="text-center"), html.Td("1.0 - 3.0", className="text-center"), html.Td("30 - 60", className="text-center"), html.Td("3 - 7", className="text-center"), html.Td("4 - 8", className="text-center"), html.Td("0 - 5", className="text-center"), html.Td("10 - 25", className="text-center")]),
                    html.Tr([html.Td("Credit Cards", className="text-center"), html.Td("3.0 - 8.0", className="text-center"), html.Td("60 - 80", className="text-center"), html.Td("1 - 3", className="text-center"), html.Td("8 - 15", className="text-center"), html.Td("40 - 70", className="text-center"), html.Td("15 - 30", className="text-center")]),
                    html.Tr([html.Td("Personal Loans", className="text-center"), html.Td("2.0 - 6.0", className="text-center"), html.Td("50 - 70", className="text-center"), html.Td("2 - 5", className="text-center"), html.Td("6 - 12", className="text-center"), html.Td("0 - 10", className="text-center"), html.Td("10 - 20", className="text-center")])
                ])
            ], bordered=True, hover=True, striped=True, className="mt-3")
        ], className="d-flex justify-content-center")
    ])

app.layout = dbc.Container([
    html.H1("CECL Model Dashboard", className="text-center my-4"),
    dbc.Row(
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Model", tab_id="model"),
                dbc.Tab(label="Model Explanation", tab_id="model-explanation")
            ], id="tabs", active_tab="model"),
            html.Div(id="tab-content", className="mt-4"),
        ], width=12, lg=10, className="mx-auto")
    )
], fluid=True, className="px-4")

@app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
def render_tab_content(active_tab):
    if active_tab == "model":
        return html.Div([
            dbc.Card([
                dbc.CardHeader(html.H5("Loan Pools", className="mb-0 text-center")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Div(["Pool Name", html.Br(), ""], className="fw-bold text-center"), width=12, md=3, className="px-1"),
                        dbc.Col(html.Div(["Balance", html.Br(), "($M)"], className="fw-bold text-center"), width=6, md=1, className="px-1"),
                        dbc.Col(html.Div(["PD", html.Br(), "(%)"], className="fw-bold text-center"), width=6, md=1, className="px-1"),
                        dbc.Col(html.Div(["LGD", html.Br(), "(%)"], className="fw-bold text-center"), width=6, md=1, className="px-1"),
                        dbc.Col(html.Div(["Term", html.Br(), "(Years)"], className="fw-bold text-center"), width=6, md=1, className="px-1"),
                        dbc.Col(html.Div(["Discount", html.Br(), "Rate (%)"], className="fw-bold text-center"), width=6, md=1, className="px-1"),
                        dbc.Col(html.Div(["Undrawn", html.Br(), "(%)"], className="fw-bold text-center"), width=6, md=1, className="px-1"),
                        dbc.Col(html.Div(["Prepayment", html.Br(), "(%)"], className="fw-bold text-center"), width=6, md=1, className="px-1"),
                    ], className="mb-2 g-0"),
                    html.Div([create_input_group(pool_id, pool_name) for pool_id, pool_name in COMMERCIAL_POOLS.items()]),
                    html.Div([create_input_group(pool_id, pool_name) for pool_id, pool_name in CONSUMER_POOLS.items()]),
                ], className="py-2"),
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader(html.H5("Economic Scenarios", className="mb-0 text-center")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Div(["Scenario", html.Br(), ""], className="fw-bold text-center"), width=12, md=3, className="px-1"),
                        dbc.Col(html.Div(["GDP Growth", html.Br(), "(%)"], className="fw-bold text-center"), width=6, md=2, className="px-1"),
                        dbc.Col(html.Div(["Unemployment", html.Br(), "(%)"], className="fw-bold text-center"), width=6, md=2, className="px-1"),
                        dbc.Col(html.Div(["Fed Funds", html.Br(), "Rate (%)"], className="fw-bold text-center"), width=6, md=2, className="px-1"),
                        dbc.Col(html.Div(["Housing Price", html.Br(), "Index"], className="fw-bold text-center"), width=6, md=2, className="px-1"),
                    ], className="mb-2 g-0"),
                    html.Div([create_economic_inputs(scenario) for scenario in ECONOMIC_SCENARIOS]),
                ], className="py-2"),
            ], className="mb-3"),
            create_weights_and_multipliers_inputs(),
            dbc.Row([
                dbc.Col(dbc.Button("Calculate", id="calculate-button", color="primary", className="me-2"), width="auto"),
                dbc.Col(dbc.Button("Reset to Defaults", id="reset-button", color="secondary"), width="auto"),
            ], className="mb-3"),
            html.Div(id="results-content"),
        ])
    elif active_tab == "model-explanation":
        return create_model_explanation()

@app.callback(
    Output("results-content", "children"),
    Input("calculate-button", "n_clicks"),
    State({"type": "pool-input", "id": ALL}, "value"),
    State({"type": "economic-input", "id": ALL}, "value"),
    State("pd-multiplier", "value"),
    State("lgd-multiplier", "value"),
    State("baseline-weight", "value"),
    State("adverse-weight", "value"),
    State("severely-adverse-weight", "value"),
)
def update_results(n_clicks, pool_inputs, economic_inputs, pd_multiplier, lgd_multiplier, baseline_weight, adverse_weight, severely_adverse_weight):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Update asset pools
    for i, pool_id in enumerate(ALL_POOLS):
        calc_engine.asset_pools[pool_id] = {
            'balance': float(pool_inputs[i*7]),
            'default-prob': float(pool_inputs[i*7 + 1]),
            'lgd': float(pool_inputs[i*7 + 2]),
            'original-term': int(float(pool_inputs[i*7 + 3])),
            'discount-rate': float(pool_inputs[i*7 + 4]),
            'undrawn-percentage': float(pool_inputs[i*7 + 5]),
            'prepayment-rate': float(pool_inputs[i*7 + 6]),
        }

    # Update economic factors
    for i, scenario in enumerate(ECONOMIC_SCENARIOS):
        calc_engine.economic_factors.loc[scenario] = {
            'gdp-growth': float(economic_inputs[i*4]),
            'unemployment-rate': float(economic_inputs[i*4 + 1]),
            'fed-funds-rate': float(economic_inputs[i*4 + 2]),
            'housing-price-index': float(economic_inputs[i*4 + 3])
        }

    # Update weights and multipliers
    calc_engine.pd_multiplier = float(pd_multiplier)
    calc_engine.lgd_multiplier = float(lgd_multiplier)
    calc_engine.scenario_weights = {
        "Baseline": float(baseline_weight),
        "Adverse": float(adverse_weight),
        "Severely Adverse": float(severely_adverse_weight)
    }

    # Normalize weights
    weight_sum = sum(calc_engine.scenario_weights.values())
    calc_engine.scenario_weights = {k: v / weight_sum for k, v in calc_engine.scenario_weights.items()}

    # Calculate ECL for each pool
    ecl_data = [(pool_id, ALL_POOLS[pool_id], calc_engine.calculate_lifetime_ecl(pool_id)) for pool_id in ALL_POOLS]
    ecl_data.sort(key=lambda x: x[2], reverse=True)
    total_reserve = sum(ecl for _, _, ecl in ecl_data)

    # Generate charts and summary
    ecl_by_pool_chart = dcc.Graph(
        figure={
            'data': [go.Bar(x=[name for _, name, _ in ecl_data], y=[ecl for _, _, ecl in ecl_data])],
            'layout': go.Layout(
                title="Lifetime ECL by Pool",
                xaxis={'title': 'Pool'},
                yaxis={'title': 'ECL ($M)'},
                height=350,
                margin=dict(l=50, r=50, t=50, b=50)
            )
        }
    )

    scenario_data = {scenario: [] for scenario in ECONOMIC_SCENARIOS}
    for pool_id in ALL_POOLS:
        for scenario in ECONOMIC_SCENARIOS:
            scenario_data[scenario].append(calc_engine.calculate_expected_loss(pool_id, scenario, 0))

    ecl_by_scenario_chart = dcc.Graph(
        figure={
            'data': [go.Bar(name=scenario, x=list(ALL_POOLS.values()), y=ecls) for scenario, ecls in scenario_data.items()],
            'layout': go.Layout(
                title="ECL by Scenario and Pool",
                xaxis={'title': 'Pool'},
                yaxis={'title': 'ECL ($M)'},
                barmode='group',
                height=350,
                margin=dict(l=50, r=50, t=50, b=50)
            )
        }
    )

    # Create a more detailed summary
    commercial_ecl = sum(ecl for pool_id, _, ecl in ecl_data if pool_id.startswith('C'))
    consumer_ecl = sum(ecl for pool_id, _, ecl in ecl_data if pool_id.startswith('P'))
    total_balance = sum(calc_engine.asset_pools[pool_id]['balance'] for pool_id in ALL_POOLS)
    
    summary_table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Category", className="text-center"),
                html.Th("Total Balance ($M)", className="text-center"),
                html.Th("ECL ($M)", className="text-center"),
                html.Th("ECL Coverage (%)", className="text-center")
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td("Commercial", className="text-center"),
                html.Td(f"{sum(calc_engine.asset_pools[pool_id]['balance'] for pool_id in COMMERCIAL_POOLS):,.2f}", className="text-center"),
                html.Td(f"{commercial_ecl:,.2f}", className="text-center"),
                html.Td(f"{commercial_ecl / sum(calc_engine.asset_pools[pool_id]['balance'] for pool_id in COMMERCIAL_POOLS) * 100:.2f}%", className="text-center")
            ]),
            html.Tr([
                html.Td("Consumer", className="text-center"),
                html.Td(f"{sum(calc_engine.asset_pools[pool_id]['balance'] for pool_id in CONSUMER_POOLS):,.2f}", className="text-center"),
                html.Td(f"{consumer_ecl:,.2f}", className="text-center"),
                html.Td(f"{consumer_ecl / sum(calc_engine.asset_pools[pool_id]['balance'] for pool_id in CONSUMER_POOLS) * 100:.2f}%", className="text-center")
            ]),
            html.Tr([
                html.Td("Total", className="fw-bold text-center"),
                html.Td(f"{total_balance:,.2f}", className="fw-bold text-center"),
                html.Td(f"{total_reserve:,.2f}", className="fw-bold text-center"),
                html.Td(f"{total_reserve / total_balance * 100:.2f}%", className="fw-bold text-center")
            ])
        ])
    ], bordered=True, hover=True, striped=True, className="mt-3")

    # Top 5 pools by ECL
    top_5_ecl = ecl_data[:5]
    top_5_table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Pool", className="text-center"),
                html.Th("ECL ($M)", className="text-center"),
                html.Th("% of Total ECL", className="text-center")
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(name, className="text-center"),
                html.Td(f"{ecl:,.2f}", className="text-center"),
                html.Td(f"{ecl / total_reserve * 100:.2f}%", className="text-center")
            ]) for _, name, ecl in top_5_ecl
        ])
    ], bordered=True, hover=True, striped=True, className="mt-3")

    ecl_summary = dbc.Card([
        dbc.CardHeader(html.H5("ECL Summary", className="mb-0 text-center")),
        dbc.CardBody([
            html.Div([summary_table], className="d-flex justify-content-center"),
            html.H6("Top 5 Pools by ECL", className="mt-3 text-center"),
            html.Div([top_5_table], className="d-flex justify-content-center")
        ], className="py-2")
    ], className="mb-3")

    return html.Div([
        dbc.Row([
            dbc.Col(ecl_by_pool_chart, md=6, className="mb-3"),
            dbc.Col(ecl_by_scenario_chart, md=6, className="mb-3"),
        ], className="mb-4"),  # Added mb-4 class to create more space below the charts
        dbc.Row([
            dbc.Col(ecl_summary, width=12)
        ], className="mt-4")  # Added mt-4 class to create more space above the ECL summary
    ])

@app.callback(
    [Output({"type": "pool-input", "id": ALL}, "value"),
     Output({"type": "economic-input", "id": ALL}, "value"),
     Output("pd-multiplier", "value"),
     Output("lgd-multiplier", "value"),
     Output("baseline-weight", "value"),
     Output("adverse-weight", "value"),
     Output("severely-adverse-weight", "value")],
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True
)
def reset_to_defaults(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    pool_default_values = [
        value
        for pool_id in ALL_POOLS
        for value in [
            DEFAULT_POOL_DATA[pool_id]['balance'],
            DEFAULT_POOL_DATA[pool_id]['default-prob'],
            DEFAULT_POOL_DATA[pool_id]['lgd'],
            DEFAULT_POOL_DATA[pool_id]['original-term'],
            DEFAULT_POOL_DATA[pool_id]['discount-rate'],
            DEFAULT_POOL_DATA[pool_id]['undrawn-percentage'],
            DEFAULT_POOL_DATA[pool_id]['prepayment-rate']
        ]
    ]

    economic_default_values = [
        value
        for scenario in ECONOMIC_SCENARIOS
        for value in [
            DEFAULT_ECONOMIC_DATA[scenario]['gdp-growth'],
            DEFAULT_ECONOMIC_DATA[scenario]['unemployment-rate'],
            DEFAULT_ECONOMIC_DATA[scenario]['fed-funds-rate'],
            DEFAULT_ECONOMIC_DATA[scenario]['housing-price-index']
        ]
    ]

    return (
        pool_default_values,
        economic_default_values,
        0.6,  # Default PD Multiplier
        0.6,  # Default LGD Multiplier
        0.4,  # Default Baseline Weight
        0.3,  # Default Adverse Weight
        0.3,  # Default Severely Adverse Weight
    )

if __name__ == '__main__':
    app.run_server(debug=True)
