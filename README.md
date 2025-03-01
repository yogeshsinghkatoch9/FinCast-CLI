# StockForecast CLI

**StockForecast CLI** is a user-friendly command-line tool for stock analysis and forecasting. It combines statistical models (ARIMA), deep learning (LSTM, GRU, Transformer-based forecasting), ensemble methods (RandomForest, XGBoost), and technical analysis libraries to offer a comprehensive solution for financial data analysis. In addition, it integrates sentiment analysis (FinBERT), model interpretability (SHAP), and interactive visualizations (Plotly).

---

## Features

1. **Data Acquisition & Preprocessing**  
   - Download historical stock data from Yahoo Finance.  
   - Clean data and handle missing values using forward-fill and interpolation.

2. **Technical Indicators**  
   - Automatically compute indicators such as SMA, EMA, MACD, RSI, Bollinger Bands, On-Balance Volume, and more.

3. **Forecasting Models**  
   - **ARIMA:** Log-transformed time series forecasting.  
   - **LSTM & GRU:** Deep learning approaches using TensorFlow/Keras.  
   - **Ensemble Methods:** RandomForest and XGBoost.  
   - **Transformer-Based Forecasting:** Advanced predictions using attention mechanisms.  
   - **Ensemble Forecasting:** Combine multiple model outputs for robust forecasting.

4. **Model Evaluation**  
   - Evaluate forecasts with RMSE, MAE, MAPE, and R².  
   - Visualize actual versus predicted values.

5. **Interactive Visualizations**  
   - Generate interactive candlestick charts and forecast plots with Plotly.

6. **Sentiment Analysis**  
   - Analyze news headlines with FinBERT to gauge market sentiment.

7. **Portfolio Management**  
   - Use simple commands to add stocks to a portfolio, view holdings, or clear the portfolio.

8. **Model Interpretability**  
   - Leverage SHAP for unified explanation graphs to interpret model predictions.

9. **Rich CLI Commands**  
   - Retrieve raw or preprocessed data, technical indicators, evaluate models, run forecasts, and more.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yogeshsinghkatoch9/StockForecast-CLI.git
cd StockForecast-CLI
