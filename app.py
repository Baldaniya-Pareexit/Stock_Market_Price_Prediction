import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import seaborn as sns
import matplotlib.dates as mdates

# Import your StockPredictor class
from stock_predictor import StockPredictor

# Set page config
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Session state initialization
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'confidence_results' not in st.session_state:
    st.session_state.confidence_results = None

# Application header
st.title('📈 Stock Price Prediction Dashboard')
st.markdown("Select a stock and time period to analyze historical data and predict future prices.")

# Create sidebar for controls
st.sidebar.header("Configuration")

# Function to get available stock files
def get_available_stocks():
    # Look for CSV files in the current directory or data folder
    # You might need to adjust the path according to your file structure
    csv_files = []
    directories = ['.', './data']
    
    for directory in directories:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith('_stock_data.csv'):
                    stock_name = file.replace('_stock_data.csv', '')
                    csv_files.append((stock_name, os.path.join(directory, file)))
    
    return csv_files

# Get available stocks
available_stocks = get_available_stocks()
stock_names = [name for name, _ in available_stocks]
stock_files = dict(available_stocks)

# If no CSV files found, provide a sample
if not available_stocks:
    st.sidebar.warning("No stock data files found. Please ensure files are named like 'STOCK_stock_data.csv'")
    stock_names = ["TCS", "RELIANCE", "HDFC"]
    stock_files = {
        "TCS": "./TCS_stock_data.csv",
        "RELIANCE": "./RELIANCE_stock_data.csv", 
        "HDFC": "./HDFC_stock_data.csv"
    }

# Stock selection
selected_stock = st.sidebar.selectbox("Select Stock", stock_names)

# Load the data
try:
    file_path = stock_files.get(selected_stock, "./TCS_stock_data.csv")
    
    # Use st.cache_data to avoid reloading the data unnecessarily
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_data(file):
        try:
            dfcopy = pd.read_csv(file)
            # Fix: Use .copy() properly
            df = dfcopy.copy()
            # Specifically convert date using the correct format
            df.columns = df.columns.str.strip().str.replace('"', '')

            # Fix: Try multiple date formats and handle conversion errors
            try:
                # First attempt with default format
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                # If too many NaT values, try specific format
                if df['Date'].isna().sum() > len(df) * 0.5:
                    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y', errors='coerce')
            except Exception as e:
                print(f"Date conversion error: {str(e)}")
                # If conversion fails, keep as string
                if 'Date' in df.columns:
                    df['Date'] = df['Date'].astype(str)

            # Clean numeric columns
            for col in df.columns:
                if col != 'Date' and col != 'series':
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace('"', '').str.replace(',', ''), errors='coerce')

            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            # Return sample data if file load fails
            return pd.DataFrame({
                'Date': pd.date_range(start='2023-01-01', periods=100),
                'series': ['EQ'] * 100,
                'OPEN': np.random.rand(100) * 500,
                'HIGH': np.random.rand(100) * 550,
                'LOW': np.random.rand(100) * 450,
                'PREV. CLOSE': np.random.rand(100) * 500,
                'ltp': np.random.rand(100) * 500,
                'close': np.random.rand(100) * 500,
                'vwap': np.random.rand(100) * 500,
                '52W H': [800] * 100,
                '52W L': [400] * 100,
                'VOLUME': np.random.randint(10000, 1000000, 100),
                'VALUE': np.random.randint(1000000, 100000000, 100),
                'No of trades': np.random.randint(1000, 10000, 100)
            })
    
    df = load_data(file_path)
    
    # Fix: Ensure Date column is properly formatted for display
    if 'Date' in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df['Date']):
            # Convert to string for display to avoid Arrow serialization issues
            df_display = df.copy()
            df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d')
        else:
            df_display = df.copy()
    else:
        df_display = df.copy()
    
    # Get min and max dates for the date selector
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        min_date = df['Date'].min()
        max_date = df['Date'].max()
    else:
        # Default date range if Date column is not properly formatted
        min_date = datetime.datetime(2020, 1, 1)
        max_date = datetime.datetime.now()
    
    # Date range selector
    st.sidebar.header("Time Period")
    
    # Predefined time periods
    period_options = {
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365,
        "3 Years": 1095,
        "All Data": None
    }
    
    selected_period = st.sidebar.selectbox("Select Predefined Period", list(period_options.keys()))
    
    # Custom date range option
    use_custom_dates = st.sidebar.checkbox("Use Custom Date Range")
    
    if use_custom_dates:
        start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Convert to datetime for filtering
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    else:
        days = period_options[selected_period]
        if days is None:
            # Use all data
            start_date = min_date
            end_date = max_date
        else:
            end_date = max_date
            start_date = end_date - pd.Timedelta(days=days)
    
    # Filter data based on selected date range
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    else:
        filtered_df = df  # Use all data if date filtering is not possible
        st.warning("Date filtering not applied due to date format issues.")
    
    # Display information about the filtered data
    st.sidebar.info(f"Selected data range: {filtered_df.shape[0]} rows")
    
    # Model parameters in sidebar
    st.sidebar.header("Model Parameters")
    lookback_days = st.sidebar.slider("Lookback Days", min_value=10, max_value=100, value=30, step=5)
    epochs = st.sidebar.slider("Training Epochs", min_value=10, max_value=200, value=50, step=10)
    future_days = st.sidebar.slider("Days to Predict", min_value=1, max_value=30, value=5, step=1)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Data Explorer", "Model Training", "Predictions"])
    
    with tab1:
        st.header(f"{selected_stock} Stock Data")
        st.write(f"Displaying data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Fix: Display the filtered dataframe with string dates to avoid Arrow serialization issues
        filtered_df_display = filtered_df.copy()
        if 'Date' in filtered_df_display.columns and pd.api.types.is_datetime64_any_dtype(filtered_df_display['Date']):
            filtered_df_display['Date'] = filtered_df_display['Date'].dt.strftime('%Y-%m-%d')
        
        # Display the dataframe with styled formatting
        st.dataframe(filtered_df_display.style.highlight_max(axis=0, subset=['HIGH']).highlight_min(axis=0, subset=['LOW']))
        
        # Show basic statistics
        st.subheader("Summary Statistics")
        # Exclude Date column from statistics
        numeric_df = filtered_df.select_dtypes(include=[np.number])
        st.dataframe(numeric_df.describe())
        
        # Basic stock price chart
        st.subheader("Stock Price History")
        
        # Check if Date is properly formatted
        if 'Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Date']):
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(filtered_df['Date'], filtered_df['close'], label='Close Price')
            ax.plot(filtered_df['Date'], filtered_df['HIGH'], label='High', alpha=0.5)
            ax.plot(filtered_df['Date'], filtered_df['LOW'], label='Low', alpha=0.5)
            
            # Format x-axis to show dates nicely
            date_format = mdates.DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_formatter(date_format)
            plt.xticks(rotation=45)
            
            ax.set_title(f"{selected_stock} Stock Price History")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab2:
        st.header("Model Training")
        st.write("Train the LSTM model on the selected data.")
        
        # Add train button
        if st.button("Train Model"):
            with st.spinner('Training model... This may take a while.'):
                try:
                    # Initialize predictor with selected parameters
                    predictor = StockPredictor(lookback_days=lookback_days)
                    
                    # Train the model
                    history, processed_data = predictor.train(filtered_df, epochs=epochs)
                    
                    # Store the trained model in session state
                    st.session_state.trained_model = predictor
                    st.session_state.processed_data = processed_data
                    
                    st.success("Model training completed!")
                    
                    # Plot training history
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(history.history['loss'], label='Training Loss')
                    ax.plot(history.history['val_loss'], label='Validation Loss')
                    ax.set_title('Model Training Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
        
        # Display model evaluation if model is trained
        if st.session_state.trained_model is not None:
            st.subheader("Model Evaluation")
            
            try:
                # Use the visualize_predictions function
                fig = plt.figure(figsize=(15, 15))
                
                # We'll call visualize_predictions but capture the output instead of showing it
                # This is a bit tricky as the original function shows plots directly
                # Instead, we'll modify it slightly to return a figure
                
                predictor = st.session_state.trained_model
                validation_split = 0.2
                
                # Data preparation
                processed_data = predictor.prepare_data(filtered_df)
                X, y = predictor.create_sequences(processed_data)
                y_pred = predictor.model.predict(X)
                
                # Get the features used during training
                features = [col for col in predictor.feature_names if col in processed_data.columns]
                
                # Scale back predictions
                dummy_array = np.zeros((len(y_pred), len(features)))
                dummy_array[:, features.index('close')] = y_pred.reshape(-1)
                predicted_prices = predictor.scaler.inverse_transform(dummy_array)[:, features.index('close')]
                
                # Scale back actual values
                dummy_array[:, features.index('close')] = y.reshape(-1)
                actual_prices = predictor.scaler.inverse_transform(dummy_array)[:, features.index('close')]
                
                # Get dates for x-axis
                dates = processed_data['Date'].iloc[predictor.lookback_days:].reset_index(drop=True)
                
                # Calculate split point
                split_idx = int(len(dates) * (1 - validation_split))
                
                # Split data into training and testing sets
                train_dates = dates[:split_idx]
                test_dates = dates[split_idx:]
                
                train_actual = actual_prices[:split_idx]
                test_actual = actual_prices[split_idx:]
                
                train_pred = predicted_prices[:split_idx]
                test_pred = predicted_prices[split_idx:]
                
                # Calculate metrics for both sets
                train_mse = mean_squared_error(train_actual, train_pred)
                train_rmse = np.sqrt(train_mse)
                train_mape = mean_absolute_percentage_error(train_actual, train_pred) * 100
                
                test_mse = mean_squared_error(test_actual, test_pred)
                test_rmse = np.sqrt(test_mse)
                test_mape = mean_absolute_percentage_error(test_actual, test_pred) * 100
                
                # Create figure with subplots
                fig = plt.figure(figsize=(15, 15))
                gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
                
                # Main price plot
                ax1 = fig.add_subplot(gs[0])
                
                # Plot training data
                ax1.plot(train_dates, train_actual, label='Training Actual', color='blue', alpha=0.7)
                ax1.plot(train_dates, train_pred, label='Training Predictions', color='red', alpha=0.7, linestyle='--')
                
                # Plot testing data
                ax1.plot(test_dates, test_actual, label='Testing Actual', color='green', alpha=0.7)
                ax1.plot(test_dates, test_pred, label='Testing Predictions', color='red', alpha=0.7, linestyle='--')
                
                # Add vertical line to show train/test split
                ax1.axvline(x=dates[split_idx], color='red', linestyle='--', alpha=0.5, label='Train/Test Split')
                
                ax1.set_title('Stock Price Predictions - Training vs Testing', pad=20, fontsize=14)
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Price')
                ax1.legend()
                ax1.grid(True)
                
                # Add metrics text
                train_metrics = (
                    f'Training Metrics:\n'
                    f'RMSE: {train_rmse:.2f}\n'
                    f'MAPE: {train_mape:.2f}%'
                )
                test_metrics = (
                    f'Testing Metrics:\n'
                    f'RMSE: {test_rmse:.2f}\n'
                    f'MAPE: {test_mape:.2f}%'
                )
                ax1.text(0.02, 0.95, train_metrics, transform=ax1.transAxes,
                         bbox=dict(facecolor='lightblue', alpha=0.8), fontsize=10)
                ax1.text(0.85, 0.95, test_metrics, transform=ax1.transAxes,
                         bbox=dict(facecolor='lightgreen', alpha=0.8), fontsize=10)
                
                # Prediction error plot
                ax2 = fig.add_subplot(gs[1])
                train_error = train_pred - train_actual
                test_error = test_pred - test_actual
                
                ax2.plot(train_dates, train_error, label='Training Error', color='blue', alpha=0.5)
                ax2.plot(test_dates, test_error, label='Testing Error', color='green', alpha=0.5)
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
                ax2.axvline(x=dates[split_idx], color='red', linestyle='--', alpha=0.5)
                ax2.set_title('Prediction Error Over Time')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Error')
                ax2.legend()
                ax2.grid(True)
                
                # Error distribution plot
                ax3 = fig.add_subplot(gs[2])
                
                # Plot training and testing error distributions separately
                sns.histplot(train_error, bins=30, alpha=0.5, color='blue', label='Training Error', ax=ax3)
                sns.histplot(test_error, bins=30, alpha=0.5, color='green', label='Testing Error', ax=ax3)
                
                ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                ax3.set_title('Error Distribution - Training vs Testing')
                ax3.set_xlabel('Error Value')
                ax3.set_ylabel('Frequency')
                ax3.legend()
                ax3.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display metrics in a more readable format
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Training Set Metrics")
                    st.metric("Root Mean Square Error (RMSE)", f"{train_rmse:.2f}")
                    st.metric("Mean Absolute % Error (MAPE)", f"{train_mape:.2f}%")
                
                with col2:
                    st.subheader("Testing Set Metrics")
                    st.metric("Root Mean Square Error (RMSE)", f"{test_rmse:.2f}")
                    st.metric("Mean Absolute % Error (MAPE)", f"{test_mape:.2f}%")
                
            except Exception as e:
                st.error(f"Error during model evaluation: {str(e)}")
    
    with tab3:
        st.header("Price Predictions")
        
        # Display disclaimer at the top of the predictions tab
        st.warning("""
        **DISCLAIMER**: The predictions shown here are based on historical data and machine learning models. 
        Past performance is not indicative of future results. These predictions should not be considered as financial advice. 
        The model does not account for unexpected market events, news, or economic changes. 
        Always conduct your own research and consult with a financial advisor before making investment decisions.
        """)
        
        if st.session_state.trained_model is None:
            st.warning("Please train a model in the 'Model Training' tab first.")
        else:
            st.write(f"Generating predictions for the next {future_days} days...")
            
            # Make predictions if not already done
            if st.session_state.predictions is None or st.button("Refresh Predictions"):
                with st.spinner('Generating predictions...'):
                    try:
                        predictor = st.session_state.trained_model
                        predictions = predictor.predict_next_days(filtered_df, days=future_days)
                        st.session_state.predictions = predictions
                        
                        # Get confidence results
                        confidence_results = predictor.analyze_prediction_confidence(
                            st.session_state.processed_data, 
                            predictions
                        )
                        st.session_state.confidence_results = confidence_results
                        
                    except Exception as e:
                        st.error(f"Error generating predictions: {str(e)}")
            
            # Show predictions
            if st.session_state.predictions is not None:
                predictions = st.session_state.predictions
                
                # Table of prediction values
                st.subheader("Forecasted Stock Prices")
                
                # Create a DataFrame for the predictions
                pred_df = pd.DataFrame({
                    'Date': predictions['dates'],
                    'Predicted Close': predictions['predictions']
                })
                
                # Format the DataFrame for display
                formatted_df = pred_df.copy()
                formatted_df['Date'] = formatted_df['Date'].dt.strftime('%Y-%m-%d')
                formatted_df['Predicted Close'] = formatted_df['Predicted Close'].round(2).astype(str).apply(lambda x: f"₹{x}")
                
                # Show the formatted DataFrame
                st.table(formatted_df)
                
                # Show prediction confidence
                if st.session_state.confidence_results is not None:
                    confidence = st.session_state.confidence_results
                    
                    st.subheader("Prediction Confidence")
                    
                    # Create gauge chart for confidence score
                    confidence_score = confidence['confidence_score'] * 100
                    
                    # Display confidence with a color-coded meter
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        # Use a progress bar as a simple meter
                        color = "red" if confidence_score < 40 else ("yellow" if confidence_score < 70 else "green")
                        st.markdown(f"### Confidence Score: {confidence_score:.1f}%")
                        st.progress(min(confidence_score/100, 1.0))
                        
                        if confidence_score < 40:
                            st.error("Low confidence in predictions - use with caution")
                        elif confidence_score < 70:
                            st.warning("Moderate confidence in predictions")
                        else:
                            st.success("High confidence in predictions")
                    
                    # Show confidence factors
                    st.subheader("Confidence Factors")
                    
                    factor_df = pd.DataFrame({
                        'Factor': confidence['factors'].keys(),
                        'Impact Score': [f"{v*100:.1f}%" for v in confidence['factors'].values()]
                    })
                    
                    st.table(factor_df)
                
                # Visualize predictions
                st.subheader("Price Forecast Visualization")
                
                # Get recent actual prices for comparison
                recent_data = filtered_df.sort_values('Date').tail(30)
                
                if 'Date' in recent_data.columns and pd.api.types.is_datetime64_any_dtype(recent_data['Date']):
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot recent actual prices
                    ax.plot(recent_data['Date'], recent_data['close'], label='Historical Close', color='blue')
                    
                    # Plot predicted prices
                    ax.plot(predictions['dates'], predictions['predictions'], label='Predicted Close', color='red', linestyle='--', marker='o')
                    
                    # Add a vertical line to separate historical data from predictions
                    last_date = recent_data['Date'].iloc[-1]
                    ax.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7)
                    
                    # Annotate the chart
                    ax.text(last_date, min(recent_data['close'].min(), predictions['predictions'].min()), 
                            'Historical | Forecast', ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8))
                    
                    # Format the chart
                    date_format = mdates.DateFormatter('%Y-%m-%d')
                    ax.xaxis.set_major_formatter(date_format)
                    plt.xticks(rotation=45)
                    
                    ax.set_title(f"{selected_stock} Stock Price Forecast")
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price')
                    ax.legend()
                    ax.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                    # Calculate potential returns
                    last_close = recent_data['close'].iloc[-1]
                    final_prediction = predictions['predictions'][-1]
                    potential_return = ((final_prediction - last_close) / last_close) * 100
                    
                    st.subheader("Potential Return Analysis")
                    
                    if potential_return > 0:
                        st.success(f"Projected {future_days}-day return: {potential_return:.2f}%")
                    else:
                        st.error(f"Projected {future_days}-day return: {potential_return:.2f}%")
                    
                    st.write(f"Last Close Price: ₹{last_close:.2f}")
                    st.write(f"Predicted Price ({predictions['dates'][-1].strftime('%Y-%m-%d')}): ₹{final_prediction:.2f}")
                    
                    # Generate investment suggestions based on predictions and confidence
                    st.subheader("Investment Suggestions")
                    
                    # Get confidence score and trend
                    confidence_score = confidence['confidence_score']
                    trend_is_up = potential_return > 0
                    trend_strength = abs(potential_return)
                    
                    # Create a box with investment suggestions
                    suggestion_box = st.container()
                    with suggestion_box:
                        # Set the background color based on the confidence
                        if confidence_score >= 0.7:
                            st.markdown("""
                            <style>
                            .high-confidence {
                                background-color: rgba(0, 200, 0, 0.1);
                                padding: 20px;
                                border-radius: 5px;
                                border-left: 5px solid green;
                            }
                            </style>
                            <div class="high-confidence">
                            """, unsafe_allow_html=True)
                        elif confidence_score >= 0.4:
                            st.markdown("""
                            <style>
                            .medium-confidence {
                                background-color: rgba(255, 165, 0, 0.1);
                                padding: 20px;
                                border-radius: 5px;
                                border-left: 5px solid orange;
                            }
                            </style>
                            <div class="medium-confidence">
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <style>
                            .low-confidence {
                                background-color: rgba(255, 0, 0, 0.1);
                                padding: 20px;
                                border-radius: 5px;
                                border-left: 5px solid red;
                            }
                            </style>
                            <div class="low-confidence">
                            """, unsafe_allow_html=True)
                        
                        # Generate recommendations based on trend and confidence
                        st.markdown("#### Key Recommendations")
                        
                        # Generate the recommendation action
                        if confidence_score >= 0.7:
                            if trend_is_up and trend_strength >= 5:
                                action = "**Strong Buy**"
                                rationale = f"High confidence prediction indicates significant upward movement of {potential_return:.2f}% over the next {future_days} days."
                            elif trend_is_up and trend_strength >= 2:
                                action = "**Buy**"
                                rationale = f"High confidence prediction suggests positive movement of {potential_return:.2f}% over the next {future_days} days."
                            elif trend_is_up:
                                action = "**Hold/Accumulate**"
                                rationale = f"High confidence prediction indicates slight upward trend of {potential_return:.2f}% over the next {future_days} days."
                            elif trend_strength >= 5:
                                action = "**Strong Sell**"
                                rationale = f"High confidence prediction indicates significant downward movement of {potential_return:.2f}% over the next {future_days} days."
                            elif trend_strength >= 2:
                                action = "**Sell**"
                                rationale = f"High confidence prediction suggests negative movement of {potential_return:.2f}% over the next {future_days} days."
                            else:
                                action = "**Hold/Reduce**"
                                rationale = f"High confidence prediction indicates slight downward trend of {potential_return:.2f}% over the next {future_days} days."
                        elif confidence_score >= 0.4:
                            if trend_is_up and trend_strength >= 5:
                                action = "**Consider Buy**"
                                rationale = f"Moderate confidence prediction suggests potential upward movement of {potential_return:.2f}% over the next {future_days} days."
                            elif trend_is_up:
                                action = "**Hold/Small Buy**"
                                rationale = f"Moderate confidence prediction indicates possible upward trend of {potential_return:.2f}% over the next {future_days} days."
                            elif trend_strength >= 5:
                                action = "**Consider Sell**"
                                rationale = f"Moderate confidence prediction suggests potential downward movement of {potential_return:.2f}% over the next {future_days} days."
                            else:
                                action = "**Hold/Watch**"
                                rationale = f"Moderate confidence prediction indicates possible downward trend of {potential_return:.2f}% over the next {future_days} days."
                        else:
                            action = "**Watch and Wait**"
                            rationale = f"Low confidence in predictions ({confidence_score*100:.1f}%) suggests holding current positions and waiting for more reliable signals."
                        
                        # Display the recommendation
                        st.markdown(f"**Suggested Action:** {action}")
                        st.markdown(f"**Rationale:** {rationale}")
                        st.markdown("---")
                        
                        # Add specific suggestions based on the prediction trend
                        st.markdown("#### Detailed Strategy")
                        
                        # Entry and exit strategies
                        if trend_is_up and confidence_score >= 0.4:
                            st.markdown("**Entry Strategy:**")
                            st.markdown("- Consider entering at market open if general market sentiment aligns with prediction")
                            st.markdown("- Set limit orders slightly below current price for better entry points")
                            st.markdown("- Consider scaling in with multiple smaller positions rather than one large order")
                            
                            st.markdown("**Exit Strategy:**")
                            st.markdown(f"- Consider setting profit targets at {min(potential_return * 1.5, 15):.2f}%")
                            st.markdown(f"- Use trailing stop-loss of {max(2.0, potential_return * 0.3):.2f}% to protect gains")
                            st.markdown("- Consider partial profit taking at key resistance levels")
                        elif not trend_is_up and confidence_score >= 0.4:
                            st.markdown("**Risk Management:**")
                            st.markdown("- Consider reducing position size if currently holding")
                            st.markdown(f"- Set stop-loss orders at {max(2.0, abs(potential_return) * 0.3):.2f}% above current price to limit losses")
                            st.markdown("- Wait for reversal signals before re-entering")
                            
                            st.markdown("**Alternative Strategies:**")
                            st.markdown("- Consider short-term hedging strategies if heavily invested")
                            st.markdown("- Look for opportunities in negatively correlated assets")
                        else:
                            st.markdown("**Recommended Approach:**")
                            st.markdown("- Monitor for additional confirmation signals before taking action")
                            st.markdown("- Wait for higher confidence predictions or clearer market trends")
                            st.markdown("- Consider focusing on other stocks with stronger signals")
                        
                        # Risk assessment
                        st.markdown("#### Risk Assessment")
                        
                        # Volatility considerations
                        if 'volatility' in confidence['factors']:
                            volatility_score = confidence['factors']['volatility']
                            if volatility_score < -0.1:
                                st.markdown("⚠️ **High Volatility Alert:** Current market conditions show elevated volatility, increasing risk of unexpected price movements.")
                            elif volatility_score > 0.1:
                                st.markdown("✅ **Low Volatility Indicator:** Current market conditions show stable price action, potentially safer for executing the strategy.")
                        
                        # Technical alignment
                        if 'technical_alignment' in confidence['factors']:
                            tech_score = confidence['factors']['technical_alignment']
                            if tech_score > 0.15:
                                st.markdown("✅ **Strong Technical Alignment:** Multiple technical indicators support the predicted direction.")
                            elif tech_score < 0.05:
                                st.markdown("⚠️ **Weak Technical Signals:** Technical indicators show mixed signals, suggesting caution.")
                        
                        # Volume support
                        if 'volume_support' in confidence['factors']:
                            volume_score = confidence['factors']['volume_support']
                            if volume_score > 0.1:
                                st.markdown("✅ **Strong Volume Support:** Trading volumes support the predicted movement direction.")
                            elif volume_score < -0.05:
                                st.markdown("⚠️ **Poor Volume Support:** Lack of trading volume may indicate weak conviction in the predicted direction.")
                        
                        # Close the div tag for styling
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Additional information and education
                        with st.expander("📚 Understanding These Recommendations"):
                            st.markdown("""
                            **How to Use These Suggestions:**
                            
                            These investment recommendations are generated algorithmically based on the model's predictions and confidence metrics. They should be used as one of many inputs in your decision-making process, not as standalone advice.
                            
                            **Key Factors in Recommendations:**
                            - **Trend Direction:** Whether prices are predicted to rise or fall
                            - **Trend Magnitude:** How significant the predicted change is
                            - **Model Confidence:** How reliable the model believes its prediction to be
                            - **Technical Factors:** Alignment with technical indicators like RSI, MACD, etc.
                            - **Volatility Assessment:** Evaluation of current market stability
                            
                            **Remember:** Even high-confidence predictions can be wrong. Always manage risk appropriately.
                            """)
                else:
                    st.warning("Cannot visualize predictions due to date formatting issues.")

except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")