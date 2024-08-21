# CECL Model Dashboard
[![Heroku Deployment](https://img.shields.io/badge/deployed-heroku-brightgreen.svg)](https://cecl-model-dashboard-db9b0de856f2.herokuapp.com/)

## Overview
The **CECL Model Dashboard** is an interactive web application designed to calculate and visualize the Current Expected Credit Loss (CECL) for various loan pools under different economic scenarios. This dashboard is built with Python using the Dash framework, incorporating Plotly for dynamic visualizations and Dash Bootstrap Components for a responsive, user-friendly interface.

### Key Features
- **Interactive Loan Pool Inputs:** Customize loan pool characteristics such as balance, probability of default (PD), loss given default (LGD), original term, discount rate, undrawn percentage, and prepayment rates.
- **Dynamic Economic Scenarios:** Adjust economic factors like GDP growth, unemployment rate, Fed funds rate, and housing price index to assess their impact on expected credit losses.
- **Weights and Multipliers:** Fine-tune the model with PD and LGD multipliers, and adjust weights for different economic scenarios.
- **Real-time CECL Calculations:** The dashboard provides immediate feedback, calculating the lifetime expected credit loss (ECL) as you modify input parameters.
- **Comprehensive Visualizations:** Explore ECL distributions across different loan pools and economic scenarios through dynamic bar charts and detailed summaries.
- **Model Explanation:** A dedicated tab provides a detailed explanation of the CECL model, including input parameters, calculation methodology, and the impact of economic factors and multipliers.

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
3. **Weights and Multipliers:** PD and LGD multipliers allow for additional risk adjustments, while scenario weights determine the importance of each economic scenario in the final calculation.
4. **Scenario Weighting:** The model applies weighted averages to the scenarios to calculate a comprehensive ECL, reflecting different potential future states of the economy.

### Visualizations
- **ECL by Loan Pool:** A bar chart showing the lifetime ECL for each loan pool.
- **ECL by Scenario and Pool:** A grouped bar chart comparing ECL across different economic scenarios for each loan pool.
- **ECL Summary:** A detailed summary table highlighting key metrics, including total balance, ECL, and ECL coverage for commercial and consumer loan pools.
- **Weights and Multipliers Summary:** A table showing the current values of PD and LGD multipliers, as well as the weights for each economic scenario.

## Getting Started
### Prerequisites
- Python 3.7 or higher
- Pip (Python package installer)

### Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/cecl-model-dashboard.git
   cd cecl-model-dashboard
   ```

2. **Create a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   python app.py
   ```

5. **Access the Dashboard:**
   Open a web browser and navigate to `http://127.0.0.1:8050/` to view the dashboard locally.

## Usage
1. **Adjust Loan Pool Inputs:** Modify the values for each loan pool, including balance, PD, LGD, and other characteristics.
2. **Set Economic Scenarios:** Adjust the economic factors for each scenario (Baseline, Adverse, Severely Adverse).
3. **Configure Weights and Multipliers:** Set the PD and LGD multipliers and adjust the weights for each economic scenario.
4. **Calculate Results:** Click the "Calculate" button to update the ECL calculations and visualizations.
5. **Explore Results:** Analyze the charts and summary tables to understand the impact of your inputs on expected credit losses.
6. **Reset to Defaults:** Use the "Reset to Defaults" button to return all inputs to their initial values.

## Contributing
Contributions to improve the CECL Model Dashboard are welcome. Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## Acknowledgments
- Dash and Plotly teams for their excellent data visualization libraries.
- The financial modeling community for insights into CECL calculation methodologies.
