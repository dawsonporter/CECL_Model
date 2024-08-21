# CECL Model Dashboard

[![Heroku Deployment](https://img.shields.io/badge/deployed-heroku-brightgreen.svg)](https://cecl-model-dashboard-db9b0de856f2.herokuapp.com/)

## Overview

The **CECL Model Dashboard** is an interactive web application designed to calculate and visualize the Current Expected Credit Loss (CECL) for various loan pools under different economic scenarios. This dashboard is built with Python using the Dash framework, incorporating Plotly for dynamic visualizations and Dash Bootstrap Components for a responsive, user-friendly interface.

### Key Features

- **Interactive Loan Pool Inputs:** Customize loan pool characteristics such as balance, probability of default (PD), loss given default (LGD), original term, and prepayment rates.
- **Dynamic Economic Scenarios:** Adjust economic factors like GDP growth, unemployment rate, Fed funds rate, and housing price index to assess their impact on expected credit losses.
- **Real-time CECL Calculations:** The dashboard provides immediate feedback, calculating the lifetime expected credit loss (ECL) as you modify input parameters.
- **Comprehensive Visualizations:** Explore ECL distributions across different loan pools and economic scenarios through dynamic bar charts and detailed summaries.
- **Model Explanation:** A dedicated tab provides a detailed explanation of the CECL model, including input parameters, calculation methodology, and the impact of economic factors.

## Live Demo

Explore the live version of the CECL Model Dashboard:

[**CECL Model Dashboard**](https://cecl-model-dashboard-db9b0de856f2.herokuapp.com/)

## Project Structure

- `app.py`: The main application file containing the Dash app, layouts, and callbacks.
- `requirements.txt`: A file listing all Python dependencies needed to run the application.
- `assets/`: Directory for custom stylesheets or additional static files.

## How It Works

### CECL Calculation Process

The dashboard calculates the lifetime expected credit losses (ECL) by considering the following factors:

1. **Loan Pool Characteristics:** Includes balance, PD, LGD, original term, discount rate, undrawn percentage, and prepayment rate.
2. **Economic Scenarios:** Users can choose from Baseline, Adverse, and Severely Adverse scenarios, each with specific economic factors like GDP growth, unemployment rate, and Fed funds rate.
3. **Scenario Weighting:** The model applies weighted averages to the scenarios to calculate a comprehensive ECL, reflecting different potential future states of the economy.

### Visualizations

- **ECL by Loan Pool:** A bar chart showing the lifetime ECL for each loan pool.
- **ECL by Scenario and Pool:** A grouped bar chart comparing ECL across different economic scenarios for each loan pool.
- **ECL Summary:** A detailed summary table highlighting key metrics, including total balance, ECL, and ECL coverage for commercial and consumer loan pools.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Pip (Python package installer)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/cecl-model-dashboard.git
   cd cecl-model-dashboard
