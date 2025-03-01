FinCast CLI

FinCast CLI is an advanced command-line tool for stock analysis and forecasting. It leverages a range of statistical models (ARIMA), deep learning techniques (LSTM, GRU, Transformer-based forecasting), ensemble methods (RandomForest, XGBoost), and technical analysis libraries to provide a comprehensive suite for financial data analysis. In addition, the tool integrates sentiment analysis using transformer models, model interpretability with SHAP, and interactive visualizations via Plotly.

Features
	•	Data Acquisition & Preprocessing:
Download historical stock data using Yahoo Finance, perform data cleaning, and handle missing values with forward-fill and interpolation.
	•	Technical Indicators:
Automatically compute a range of technical indicators such as SMA, EMA, MACD, RSI, Bollinger Bands, On-Balance Volume, volatility, and more.
	•	Forecasting Models:
Forecast future stock prices using multiple methods:
	•	ARIMA: Log-transformed time series forecasting.
	•	LSTM & GRU: Deep learning approaches using TensorFlow.
	•	Ensemble Methods: RandomForest and XGBoost.
	•	Transformer-Based Forecasting: Leverage attention mechanisms for advanced forecasting.
	•	Ensemble Forecasting: Combine predictions from multiple models to form a robust forecast.
	•	Model Evaluation:
Evaluate forecasting performance with metrics such as RMSE, MAE, MAPE, and R². Visualize the forecast versus actual data.
	•	Interactive Visualizations:
Generate interactive candlestick charts and forecast visualizations using Plotly.
	•	Sentiment Analysis:
Analyze news headlines with FinBERT to gauge market sentiment.
	•	Portfolio Management:
Simple commands to add stocks to a portfolio, view holdings, and clear the portfolio.
	•	Unified Explanations:
Use SHAP to generate a unified explanation graph for model interpretability.
	•	CLI Commands:
A rich set of commands lets you quickly get raw data, preprocessed data, technical features, perform evaluation, generate forecasts, and more. See the commands below for details.

Installation
	1.	Clone the Repository:

git clone https://github.com/your_username/FinCast-CLI.git
cd FinCast-CLI


	2.	Create a Virtual Environment (Optional but Recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


	3.	Install Dependencies:
Make sure you have Python 3.7+ installed. Then run:

pip install -r requirements.txt

(Note: Your requirements.txt should list dependencies such as yfinance, numpy, pandas, matplotlib, statsmodels, arch, tensorflow, scikit-learn, xgboost, transformers, shap, ta, plotly, etc.)

Usage

Run the CLI tool by executing:

python your_script.py

Once running, you will see a prompt. Type help to see the list of available commands. Some examples include:
	•	Data Inspection:
	•	raw TICKER – Display the first and last 5 rows of raw data.
	•	preprocessed TICKER – Show cleaned data.
	•	features TICKER – Display technical indicators computed on the data.
	•	Forecasting:
	•	oneday TICKER – 1-day ensemble forecast.
	•	arima TICKER – 1-day forecast using ARIMA.
	•	lstm TICKER – 1-day forecast using an LSTM model.
	•	predict TICKER DAYS [MODEL] – Forecast for a custom horizon (default model is LSTM).
	•	final TICKER DAYS [MODEL] – Run a full analysis: model evaluation, forecast printout, interactive candlestick and forecast charts, plus a unified SHAP explanation.
	•	Visualization:
	•	candlestick TICKER [PERIOD] [INTERVAL] – Show an interactive candlestick chart.
	•	interactive TICKER DAYS [MODEL] – Launch an interactive forecast visualization.
	•	Stock Information:
	•	info TICKER – Get basic stock information from Yahoo Finance.
	•	analyze TICKER – Run a comprehensive analysis including technical indicators and forecast.
	•	yahoo TICKER – Display a Yahoo Finance–like dashboard.
	•	Sentiment Analysis:
	•	sentiment HEADLINE – Analyze the sentiment of a news headline.
	•	Portfolio Management:
	•	portfolio add TICKER SHARES – Add shares of a stock to your portfolio.
	•	portfolio show – Display your current portfolio.
	•	portfolio clear – Clear the portfolio.
	•	Other:
	•	help – Display a help message with all commands.
	•	exit – Quit the CLI.

Code Structure
	•	Data Functions:
Download and preprocess stock data; create sequences for deep learning models.
	•	Forecast Functions:
Functions for each forecasting model (ARIMA, LSTM, GRU, RF, XGB, Transformer, and Ensemble).
	•	Visualization Functions:
Functions to plot multi-day forecasts, interactive candlestick charts, and forecast charts.
	•	CLI Command Parser:
A command loop to accept user inputs and run corresponding functions.
	•	Portfolio Management:
Simple commands to manage a portfolio of stocks.

Customization

Feel free to extend the CLI by adding more models, visualizations, or additional analysis features. Contributions and suggestions are welcome!
