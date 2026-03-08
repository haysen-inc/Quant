import yfinance as yf
import pandas as pd
import os

def fetch_spy_data(start_date="2000-01-01", end_date=None, interval="1d", save_dir="data"):
    """
    Fetches historical SPY data using yfinance and saves it to a CSV file.
    Supports multi-timeframe intraday intervals like 15m, 1h, 1d.
    
    Args:
        start_date (str): Start date for data in 'YYYY-MM-DD' format.
        end_date (str): End date for data in 'YYYY-MM-DD' format. Default is today.
        interval (str): Data interval (valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo).
        save_dir (str): Directory to save the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing historical SPY data.
    """
    print(f"Fetching SPY data ({interval}) from {start_date} to {end_date if end_date else 'today'}...")
    spy = yf.Ticker("SPY")
    
    # yfinance has limitations on intraday historical data (e.g. 1h is max 730 days, 15m is max 60 days)
    try:
        df = spy.history(start=start_date, end=end_date, interval=interval)
    except Exception as e:
        print(f"Error fetching {interval} data: {e}. Note: Yahoo limits intraday data history.")
        return None
        
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"SPY_{interval}_historical.csv")
    df.to_csv(save_path)
    print(f"Data successfully saved to {save_path}")
    print(f"Total rows: {len(df)}")
    
    return df

if __name__ == "__main__":
    df = fetch_spy_data()
    print(df.head())
