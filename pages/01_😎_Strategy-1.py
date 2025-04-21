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
        utc = pytz.UTC
        
        # Fetch stock data
        stock_data = yf.Ticker(ticker)
        stock_data_hist = stock_data.history(period=period, interval=interval)
        
        # Ensure the index is timezone-aware
        if stock_data_hist.index.tz is None:
            # If not timezone-aware, assume UTC
            stock_data_hist.index = stock_data_hist.index.tz_localize(utc)
        
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
    
    # Sidebar for parameters
    st.sidebar.header("Analysis Parameters")
    
    # Get current time in IST
    ist = pytz.timezone('Asia/Kolkata')
    utc = pytz.UTC
    current_time = pd.Timestamp.now(tz=ist)
    
    # Start datetime selection
    st.sidebar.subheader("Select Time Range")
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=current_time.date() - pd.Timedelta(days=1),
        max_value=current_time.date()
    )
    
    start_time = st.sidebar.time_input(
        "Start Time",
        value=pd.Timestamp('09:00').time()
    )
    
    # End datetime selection
    end_date = st.sidebar.date_input(
        "End Date",
        value=current_time.date(),
        min_value=start_date,
        max_value=current_time.date()
    )
    
    end_time = st.sidebar.time_input(
        "End Time",
        value=current_time.time()
    )
    
    # Combine date and time and convert to IST
    start_datetime_ist = pd.Timestamp.combine(start_date, start_time).tz_localize(ist)
    end_datetime_ist = pd.Timestamp.combine(end_date, end_time).tz_localize(ist)
    
    # Convert to UTC for data fetching
    start_datetime_utc = start_datetime_ist.tz_convert(utc)
    end_datetime_utc = end_datetime_ist.tz_convert(utc)

    # Multiple Stock Selection
    selected_assets = st.sidebar.multiselect(
        "Select Assets",
        options=list(stock_data['dict'].keys()),
        default=list(stock_data['dict'].keys())[:3]  # Default to first 3 assets
    )

    # Define available intervals
    interval_options = ['1m', '5m', '15m', '30m', '60m', '4h', '1d']


    # Multiple Interval Selection
    selected_intervals = st.sidebar.multiselect(
        "Select Intervals",
        options=interval_options,
        default=['5m']
    )

    # Auto-refresh option
    st.sidebar.write("Auto-refresh options")
    auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
    
    if auto_refresh:
        refresh_interval = st.sidebar.selectbox(
            "Refresh Interval", 
            options=[30, 60, 120, 300],
            format_func=lambda x: f"{x} seconds"
        )
        
        # Add a timer to trigger refresh
        st.sidebar.write(f"Will refresh in {refresh_interval} seconds")
        
        # Use JavaScript to implement auto-refresh
        st.markdown(f"""
        <script>
        setTimeout(function() {{
            window.location.reload();
        }}, {refresh_interval * 1000});
        </script>
        """, unsafe_allow_html=True)


    # Create a combined list of asset-interval pairs
    asset_interval_pairs = [
        (asset, interval) 
        for asset in selected_assets 
        for interval in selected_intervals
    ]

    # Summary containers for overall analysis
    summary_container = st.container()
    detail_container = st.container()

    # Prepare summary data
    summary_data = []

    # Process each asset-interval pair
    with detail_container:
        for selected_asset, selected_interval in asset_interval_pairs:
            try:
                # Get ticker for the selected asset
                selected_stock = stock_data['dict'][selected_asset]
                
                # Create section header
                st.markdown(f"""
                    <div style='background-color:#f0f2f6; padding:10px; border-radius:5px;'>
                    <h3>🔍 Analysis: {selected_asset} ({selected_stock}) - {selected_interval}</h3>
                    </div>
                """, unsafe_allow_html=True)

                # Fetch stock data
                df = fetch_stock_data(selected_stock, period='5d', interval=selected_interval)
                
                if df is not None and not df.empty:
                    # Filter data based on UTC datetime range
                    df = df[(df.index >= start_datetime_utc) & (df.index <= end_datetime_utc)]
                    
                    if len(df) > 0:
                        # Process data
                        df = calculate_bollinger_bands(df)
                        df, alert_message = create_bollinger_count_columns(df)

                        # Display alerts if any
                        if alert_message:
                            st.markdown(alert_message, unsafe_allow_html=True)

                        # Separate Upper and Lower Break DataFrames
                        df_upper = df[df['count_bullish_upper_break'] == 1]
                        df_lower = df[df['count_bullish_lower_break'] == 1]
                        
                        # Calculate Cumulative Counters
                        df_upper = create_advanced_bollinger_cumulative_counters_upper(df_upper, 'count_bullish_upper_break')
                        df_lower = create_advanced_bollinger_cumulative_counters_lower(df_lower, 'count_bullish_lower_break')
                        
                        # Combine DataFrames
                        df_final = pd.concat([df, df_upper, df_lower]).sort_index()
                        
                        # Create Reversal Report
                        reversal_report = create_reversal_report(df_final, selected_stock, selected_interval)
                        
                        # Display time range
                        st.markdown(f"""
                            <p style='color: #666; font-size: 14px;'>
                            Time Range (IST): {start_datetime_ist.strftime('%Y-%m-%d %H:%M')} to {end_datetime_ist.strftime('%Y-%m-%d %H:%M')}
                            </p>
                        """, unsafe_allow_html=True)
                        
                        # Display report
                        st.dataframe(reversal_report, use_container_width=True)
                        
                        # Prepare summary data
                        summary_data.append({
                            'Asset': selected_asset,
                            'Interval': selected_interval,
                            'Upper Breaks': reversal_report[reversal_report['Direction'] == 'Up'].shape[0],
                            'Lower Breaks': reversal_report[reversal_report['Direction'] == 'Down'].shape[0]
                        })
                    
                    else:
                        st.warning(f"No data available for {selected_asset} in the selected time range")
                
                else:
                    st.warning(f"Failed to fetch data for {selected_asset}")
                
                # Separator between assets
                st.markdown("---")

            except Exception as e:
                st.error(f"Error processing {selected_asset}: {str(e)}")

    # Display Summary
    with summary_container:
        st.header("Analysis Summary")
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No summary data available")

if __name__ == "__main__":
    main()