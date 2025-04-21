import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz

# Configure the page settings for full width
st.set_page_config(
    page_title="Bollinger Bands Alert System",
    page_icon="📊",
    layout="wide",  # Set layout to wide for full-width coverage
)


# Add this after your imports
st.markdown("""
<style>
.alert {
    padding: 20px;
    background-color: #4CAF50;
    color: white;
    margin-bottom: 15px;
    opacity: 1;
    transition: opacity 0.6s;
}

.alert.success {
    background-color: #4CAF50;
}

.closebtn {
    margin-left: 15px;
    color: white;
    font-weight: bold;
    float: right;
    font-size: 22px;
    line-height: 20px;
    cursor: pointer;
    transition: 0.3s;
}
</style>
""", unsafe_allow_html=True)

# Add this JavaScript for sound
st.markdown("""
<script>
function playAlertSound() {
    var audio = new Audio('https://actions.google.com/sounds/v1/alarms/beep_short.ogg');
    audio.play();
}
</script>
""", unsafe_allow_html=True)


def load_stock_list(csv_path='assets.csv'):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Create a dictionary of assets and tickers
        stock_dict = dict(zip(df['Asset'], df['ticker']))
        
        return {
            'tickers': df['ticker'].tolist(),
            'dict': stock_dict
        }
    
    except Exception as e:
        st.error(f"Error reading CSV: {e}. Using default list.")
        return {
            'tickers': ['EURUSD=X', 'INFY.NS', 'GC=F'],
            'dict': {
                'EURO': 'EURUSD=X', 
                'Infosys': 'INFY.NS', 
                'Gold': 'GC=F'
            }
        }


def fetch_stock_data(ticker, period, interval):
    try:
        # Set timezone to IST
        ist = pytz.timezone('Asia/Kolkata')
        
        # Fetch stock data
        stock_data = yf.Ticker(ticker)
        stock_data_hist = stock_data.history(period=period, interval=interval)
        
        # If the index is not timezone-aware, localize to UTC first
        if stock_data_hist.index.tz is None:
            stock_data_hist.index = stock_data_hist.index.tz_localize('UTC')
        
        # Convert to IST
        stock_data_hist.index = stock_data_hist.index.tz_convert(ist)
        
        # Select required columns
        df = stock_data_hist[['Open', 'Close', 'Low', 'High']]
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_bollinger_bands(df, timeperiod=20, nbdevup=2, nbdevdn=2):
    # Initialize Bollinger Bands indicator
    bollinger = ta.volatility.BollingerBands(
        close=df['Close'],
        window=timeperiod,
        window_dev=nbdevup  # Both up and down standard deviations will use this value
    )
    
    # Calculate Bollinger Bands
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    df['BB_Lower'] = bollinger.bollinger_lband()
    
    return df


def create_bollinger_count_columns(df):
    df['count_bullish_lower_break'] = 0
    df['count_bullish_upper_break'] = 0

    bullish_mask = df['Open'] < df['Close']
    bearish_mask = df['Open'] > df['Close']

    df.loc[bullish_mask & (df['Close'] < df['BB_Lower']), 'count_bullish_lower_break'] = 1
    df.loc[bullish_mask & (df['Close'] > df['BB_Upper']), 'count_bullish_upper_break'] = 1
    df.loc[bearish_mask & (df['Close'] < df['BB_Lower']), 'count_bullish_lower_break'] = 1
    df.loc[bearish_mask & (df['Close'] > df['BB_Upper']), 'count_bullish_upper_break'] = 1

    total_bullish = df['count_bullish_upper_break'].sum()
    total_bearish = df['count_bullish_lower_break'].sum()

    alert_message = None
    if total_bullish > 0 or total_bearish > 0:
        alert_message = f"""
        <div class="alert success">
            <span class="closebtn" onclick="this.parentElement.style.display='none';">&times;</span>
            🔔 ALERT: {total_bullish} upper breaks, {total_bearish} lower breaks detected!
        </div>
        <audio autoplay>
            <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
        </audio>
        """

    return df, alert_message


def create_advanced_bollinger_cumulative_counters_upper(df, break_column):
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Initialize cumulative counter columns
    df['cumulative_upper_band_counter'] = 1
    
    # Upper Band Cumulative Counter (only when count_bullish_upper_break = 1)
    for i in range(len(df) - 1):
        # Check if current row has upper break
        if df.loc[df.index[i], 'count_bullish_upper_break'] == 1:
            # 2 Bullish Candle (Open < Close)
            if (df.loc[df.index[i], 'Open'] < df.loc[df.index[i], 'Close']) and \
                (df.loc[df.index[i+1], 'Open'] < df.loc[df.index[i+1], 'Close']) and \
               (df.loc[df.index[i+1], 'Close'] > df.loc[df.index[i], 'Close']):
                df.loc[df.index[i+1], 'cumulative_upper_band_counter'] = 1
            # 1 bullish (Open < Close) and adjacent 1 Bearish Candle (Open > Close)
            elif (df.loc[df.index[i], 'Open'] < df.loc[df.index[i], 'Close']) and \
                (df.loc[df.index[i+1], 'Open'] > df.loc[df.index[i+1], 'Close']) and \
               (df.loc[df.index[i+1], 'Close'] > df.loc[df.index[i], 'Close']):
                df.loc[df.index[i+1], 'cumulative_upper_band_counter'] = 1
            # 2 Bearish Candle (Open > Close)
            elif (df.loc[df.index[i], 'Open'] > df.loc[df.index[i], 'Close']) and \
                (df.loc[df.index[i+1], 'Open'] > df.loc[df.index[i+1], 'Close']) and \
                 (df.loc[df.index[i+1], 'Close'] > df.loc[df.index[i], 'Close']):
                df.loc[df.index[i+1], 'cumulative_upper_band_counter'] = 1
             # 1 Bearish Candle (Open > Close) and 1 bullish candle ((Open < Close))
            elif (df.loc[df.index[i], 'Open'] > df.loc[df.index[i], 'Close']) and \
                (df.loc[df.index[i+1], 'Open'] < df.loc[df.index[i+1], 'Close']) and \
                 (df.loc[df.index[i+1], 'Close'] > df.loc[df.index[i], 'Close']):
                df.loc[df.index[i+1], 'cumulative_upper_band_counter'] = 1
            else:
                df.loc[df.index[i+1], 'cumulative_upper_band_counter'] = -1
    
    # Calculate cumulative sum
    df['cumulative_upper_band_total'] = df['cumulative_upper_band_counter'].cumsum()
    
    return df

def create_advanced_bollinger_cumulative_counters_lower(df, break_column):
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Initialize cumulative counter columns
    df['cumulative_lower_band_counter'] = 1
    
   
    # Lower Band Cumulative Counter (only when count_bullish_lower_break = 1)
    for i in range(len(df) - 1):
        # Check if current row has lower break
        if df.loc[df.index[i], 'count_bullish_lower_break'] == 1:
            # 2 Bullish Candle (Open < Close)
            if (df.loc[df.index[i], 'Open'] < df.loc[df.index[i], 'Close']) and \
                (df.loc[df.index[i+1], 'Open'] < df.loc[df.index[i+1], 'Close']) and \
               (df.loc[df.index[i+1], 'Close'] < df.loc[df.index[i], 'Close']):
                df.loc[df.index[i+1], 'cumulative_lower_band_counter'] = 1
            # 1 Bullish Candle (Open < Close) and 1 Bearish Candle (Open > Close)
            elif (df.loc[df.index[i], 'Open'] < df.loc[df.index[i], 'Close']) and \
                (df.loc[df.index[i+1], 'Open'] > df.loc[df.index[i+1], 'Close']) and \
               (df.loc[df.index[i+1], 'Close'] < df.loc[df.index[i], 'Close']):
                df.loc[df.index[i+1], 'cumulative_lower_band_counter'] = 1
            # 2 Bearish Candle (Open > Close)
            elif (df.loc[df.index[i], 'Open'] > df.loc[df.index[i], 'Close']) and \
                (df.loc[df.index[i+1], 'Open'] > df.loc[df.index[i+1], 'Close']) and \
                 (df.loc[df.index[i+1], 'Close'] < df.loc[df.index[i], 'Close']):
                df.loc[df.index[i+1], 'cumulative_lower_band_counter'] = 1
            # 1 Bearish Candle (Open > Close) and 1 Bullish Candle (Open < Close)
            elif (df.loc[df.index[i], 'Open'] > df.loc[df.index[i], 'Close']) and \
                (df.loc[df.index[i+1], 'Open'] < df.loc[df.index[i+1], 'Close']) and \
                 (df.loc[df.index[i+1], 'Close'] < df.loc[df.index[i], 'Close']):
                df.loc[df.index[i+1], 'cumulative_lower_band_counter'] = 1
            else:
                df.loc[df.index[i+1], 'cumulative_lower_band_counter'] = -1
    
    # Calculate cumulative sum
    df['cumulative_lower_band_total'] = df['cumulative_lower_band_counter'].cumsum()
    
    return df

def create_reversal_report(df, ticker, interval):
    upper_breaks = df[df['count_bullish_upper_break'] == 1]
    lower_breaks = df[df['count_bullish_lower_break'] == 1]
    
    report_data = []
    
    # Process upper breaks
    for idx, row in upper_breaks.iterrows():
        idx = pd.to_datetime(idx)  # Ensure idx is a datetime object
        cumulative_count = row.get('cumulative_upper_band_total', None)
        if cumulative_count is not None:  # Only add if there's a cumulative count
            report_data.append({
                'Pair': ticker,
                'Direction': 'Up',
                'Time Frame/Interval': interval,
                'Cumulative Count': cumulative_count,
                'Time': idx.strftime('%H:%M'),
                'Date': idx.strftime('%B %d, %Y'),
                'Price': row['Close'],
                'Timestamp': idx  # Add full timestamp for sorting
            })
    
    # Process lower breaks
    for idx, row in lower_breaks.iterrows():
        idx = pd.to_datetime(idx)  # Ensure idx is a datetime object
        cumulative_count = row.get('cumulative_lower_band_total', None)
        if cumulative_count is not None:  # Only add if there's a cumulative count
            report_data.append({
                'Pair': ticker,
                'Direction': 'Down',
                'Time Frame/Interval': interval,
                'Cumulative Count': cumulative_count,
                'Time': idx.strftime('%H:%M'),
                'Date': idx.strftime('%B %d, %Y'),
                'Price': row['Close'],
                'Timestamp': idx  # Add full timestamp for sorting
            })
    
    # Convert to DataFrame
    reversal_report = pd.DataFrame(report_data)
    
    if not reversal_report.empty:
        # Sort by timestamp and get the highest cumulative count for each timestamp
        reversal_report = (reversal_report.sort_values('Cumulative Count', ascending=False)
                         .drop_duplicates(subset=['Time', 'Direction'], keep='first')
                         .sort_values('Timestamp'))
    
    # Drop the Timestamp column before displaying
    if not reversal_report.empty:
        reversal_report = reversal_report.drop(columns=['Timestamp'])
    
    return reversal_report


def advanced_bollinger_cumulative_plot(df):
    fig = make_subplots(
        rows=3, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        subplot_titles=(
            'Candlestick with Bollinger Bands', 
            'Upper Band Cumulative Counter', 
            'Lower Band Cumulative Counter'
        ),
        row_heights=[0.5, 0.25, 0.25]
    )

    # Candlestick
    candlestick = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    )
    fig.add_trace(candlestick, row=1, col=1)

    # Bollinger Bands
    upper_band = go.Scatter(
        x=df.index, 
        y=df['BB_Upper'], 
        mode='lines', 
        name='Upper Band',
        line=dict(color='green', width=1)
    )
    fig.add_trace(upper_band, row=1, col=1)

    lower_band = go.Scatter(
        x=df.index, 
        y=df['BB_Lower'], 
        mode='lines', 
        name='Lower Band',
        line=dict(color='green', width=1)
    )
    fig.add_trace(lower_band, row=1, col=1)

    # Upper Band Cumulative Counter
    upper_break_df = df[df['count_bullish_upper_break'] == 1]
    upper_counter = go.Scatter(
        x=upper_break_df.index,
        y=upper_break_df['cumulative_upper_band_total'],
        mode='markers',
        name='Upper Band Cumulative Total',
        line=dict(color='blue', width=2),
        marker=dict(color='blue', size=8)
    )
    fig.add_trace(upper_counter, row=2, col=1)

    # Lower Band Cumulative Counter
    lower_break_df = df[df['count_bullish_lower_break'] == 1]
    lower_counter = go.Scatter(
        x=lower_break_df.index,
        y=lower_break_df['cumulative_lower_band_total'],
        mode='markers',
        name='Lower Band Cumulative Total',
        line=dict(color='red', width=2),
        marker=dict(color='red', size=8)
    )
    fig.add_trace(lower_counter, row=3, col=1)

    # Update layout
    fig.update_layout(
        title='Advanced Bollinger Bands Cumulative Counters',
        height=1000,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Upper Band Counter", row=2, col=1)
    fig.update_yaxes(title_text="Lower Band Counter", row=3, col=1)

    return fig

def main():
    st.title("Bollinger Bands Alert Analysis")
    
    # Load stock list
    stock_data = load_stock_list()
    
    # Sidebar for stock and period selection
    st.sidebar.header("Analysis Parameters")
    
    # Stock Selection
    selected_asset = st.sidebar.selectbox("Select Asset", list(stock_data['dict'].keys()))
    selected_stock = stock_data['dict'][selected_asset]

    # Define available intervals for 5 days period
    interval_options = ['1m', '5m', '15m', '30m', '60m', '4h', '1d']

    # Fixed period (not shown in sidebar)
    selected_period = '5d'

    # Interval Selection
    selected_interval = st.sidebar.selectbox("Select Interval", interval_options)

    # Fetch and Process Data
    try:
        df = fetch_stock_data(selected_stock, selected_period, selected_interval)
        
        if df is not None and not df.empty:
            # Calculate Bollinger Bands
            df = calculate_bollinger_bands(df)
            
            # Create Bollinger Break Columns
            #df = create_bollinger_count_columns(df)
            df, alert_message = create_bollinger_count_columns(df)

            if alert_message:
                st.markdown(alert_message, unsafe_allow_html=True)
                       
            # Separate Upper and Lower Break DataFrames
            df_upper = df[df['count_bullish_upper_break'] == 1]
            df_lower = df[df['count_bullish_lower_break'] == 1]
            
            # Calculate Cumulative Counters
            df_upper = create_advanced_bollinger_cumulative_counters_upper(df_upper, 'count_bullish_upper_break')
            df_lower = create_advanced_bollinger_cumulative_counters_lower(df_lower, 'count_bullish_lower_break')
            
            # Combine DataFrames
            df_final = pd.concat([df, df_upper, df_lower])
            df_final = df_final.sort_index()
            
            # Plot Cumulative Bollinger Bands
            fig = advanced_bollinger_cumulative_plot(df_final)
            st.plotly_chart(fig)
            
            # Create Reversal Report (now sorted by time)
            reversal_report = create_reversal_report(df_final, selected_stock, selected_interval)
            
            # Display Reversal Report
            st.subheader("Alert")
            st.dataframe(reversal_report)
            
            # Download Button for Reversal Report
            csv = reversal_report.to_csv(index=False)
            st.download_button(
                label="Download Report",
                data=csv,
                file_name=f'{selected_stock}_report.csv',
                mime='text/csv'
            )
        else:
            st.warning("No data available for the selected parameters.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()