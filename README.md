# **📈 Stock Market Price Prediction with LSTM**

This project is a stock market price prediction tool using Long Short-Term Memory (LSTM) neural networks built with TensorFlow/Keras. It includes technical indicator calculations, future price forecasting, and confidence analysis—all in a streamlined, Python-based web interface powered by Streamlit.


# 🚀 **Features:**

📊 Technical Analysis: Calculates indicators like RSI, MACD, SMA, Bollinger Bands, etc.

🔮 Forecasting: Predicts future stock closing prices using an LSTM model.

✅ Confidence Scoring: Evaluates prediction confidence based on market volatility, trend strength, volume, and technical alignment.

🌐 Interactive Interface: Upload CSV data and get predictions directly via the Streamlit web app.


# 🧠 **Model:**

Architecture: 2 LSTM layers with dropout, followed by dense layers.

Input: Sequences of the past n days (default: 30) of technical indicators.

Output: Predicted closing price for upcoming days (default: 5).

Training: Uses early stopping on validation loss.


# 📦 **Installation:**

Clone the repo:

git clone https://github.com/yourusername/stock-price-predictor.git

cd stock-price-predictor

Install dependencies:

pip install -r requirements.txt

Note: You may need tensorflow, pandas, scikit-learn, numpy, and streamlit.


# 🖥️ **Usage:**

Run the app:

streamlit run app.py

Interface allows you to:

Upload your stock data CSV file

View processed charts and predictions

See a 5-day forecast and confidence levels


# 📂 **File Structure:**

app.py: Streamlit frontend to interact with the model

stock_predictor.py: Core logic for data preprocessing, model training, and forecasting

README.md: Project documentation

requirements.txt: List of required Python packages (to be added manually)


# 🧪 **Sample Input Format:**

Your CSV should include at least the following columns:

Date, OPEN, HIGH, LOW, close, VOLUME, vwap

Dates should ideally be in YYYY-MM-DD format, but flexible parsing is supported.


# 📈 **Example Output:**

Predicted Price: ₹1234.56

Prediction Confidence: 78%

Factors:

📉 Low volatility

📈 Positive trend

📊 High volume

✅ Bullish technicals


# ✅ **TODO:**

Add support for multiple stock symbols

Optimize hyperparameters

Add time series visualization


# 📃 **License:**

This project is licensed under the MIT License.
