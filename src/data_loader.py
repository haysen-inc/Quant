import yfinance as yf
import pandas as pd
import os


def fetch_spy_data(start_date="2000-01-01", end_date=None, interval="1d", save_dir="data"):
    """
    Fetches historical SPY data using yfinance, with append-only strategy
    for intraday data to keep the start date fixed (stable expert baseline).

    For intraday intervals (1h, 15m, etc.):
      - If saved CSV exists, only fetch new bars after the last saved timestamp
      - Append new bars to the CSV (never drop old data)
      - This ensures indicators computed on old data never change

    For daily data: always fetch fresh (no 730-day yfinance limit issue).
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"SPY_{interval}_historical.csv")

    is_intraday = interval in ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h')

    if is_intraday and os.path.exists(save_path):
        # Load existing data
        existing = pd.read_csv(save_path, index_col=0)
        existing.index = pd.to_datetime(existing.index, utc=True)
        last_ts = existing.index[-1]
        # Fetch only new bars after last saved timestamp
        fetch_start = (last_ts - pd.Timedelta(hours=2)).strftime('%Y-%m-%d')
        print(f"Appending SPY data ({interval}) from {fetch_start} to {end_date or 'now'}...")
        try:
            spy = yf.Ticker("SPY")
            new_df = spy.history(start=fetch_start, end=end_date, interval=interval)
        except Exception as e:
            print(f"Fetch failed: {e}, using cached data")
            return existing

        if new_df is not None and len(new_df) > 0:
            # Only keep bars strictly after the last existing timestamp
            new_bars = new_df[new_df.index > last_ts]
            if len(new_bars) > 0:
                # Align columns
                common_cols = existing.columns.intersection(new_bars.columns)
                combined = pd.concat([existing[common_cols], new_bars[common_cols]])
                combined = combined[~combined.index.duplicated(keep='first')]
                combined.sort_index(inplace=True)
                combined.to_csv(save_path)
                print(f"Appended {len(new_bars)} new bars (total: {len(combined)})")
                return combined
            else:
                print(f"No new bars (total: {len(existing)})")
                return existing
        else:
            print(f"No data returned, using cached ({len(existing)} bars)")
            return existing
    else:
        # Fresh fetch (first time or daily data)
        print(f"Fetching SPY data ({interval}) from {start_date} to {end_date if end_date else 'today'}...")
        spy = yf.Ticker("SPY")
        try:
            df = spy.history(start=start_date, end=end_date, interval=interval)
        except Exception as e:
            print(f"Error fetching {interval} data: {e}")
            return None

        df.to_csv(save_path)
        print(f"Data saved to {save_path}")
        print(f"Total rows: {len(df)}")
        return df


if __name__ == "__main__":
    df = fetch_spy_data()
    print(df.head())
