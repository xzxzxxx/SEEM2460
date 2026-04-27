import yfinance as yf
import pandas as pd
import os

def fetch_crypto_data():
    """
    Fetch historical daily close prices for major cryptocurrencies using Yahoo Finance.
    We select a basket of top coins by market cap (excluding stablecoins if possible).
    """
    
    # List of major cryptocurrency tickers on Yahoo Finance
    tickers = [
        "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", 
        "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "LTC-USD",
        "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD", "UNI7083-USD",
        "XMR-USD", "ETC-USD", "BCH-USD", "XLM-USD", "ALGO-USD"
    ]
    
    start_date = "2021-01-01"
    end_date = "2024-01-01"
    
    print(f"Downloading data for {len(tickers)} cryptocurrencies from {start_date} to {end_date}...")
    
    # Use yfinance to download daily data
    # yfinance recent versions auto-adjust prices and use 'Close' instead of 'Adj Close'
    # We set multi_level_index=False to ensure flat column names if using version >= 0.2.50
    try:
        df = yf.download(tickers, start=start_date, end=end_date, multi_level_index=False)['Close']
    except TypeError:
        # Fallback for older yfinance versions that don't support multi_level_index
        df = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Clean the column names (remove '-USD' suffix)
    # yf.download might return columns with ticker names directly
    df.columns = [str(col).replace('-USD', '') for col in df.columns]
    
    # Check for missing values and forward-fill small gaps (e.g., weekend trading glitches)
    df = df.ffill()
    
    # Drop columns (coins) that have missing values for more than 20% of the timeline
    df = df.dropna(axis=1, thresh=int(len(df) * 0.8))
    
    # Drop any remaining rows with missing values to ensure a clean correlation matrix later
    df = df.dropna()
    
    os.makedirs('data', exist_ok=True)
    output_path = 'data/crypto_prices.csv'
    df.to_csv(output_path)
    
    print(f"\nData successfully downloaded and saved to {output_path}")
    print(f"Final dataset shape: {df.shape} (Rows: Dates, Columns: Coins)")
    
    return df

if __name__ == "__main__":
    fetch_crypto_data()