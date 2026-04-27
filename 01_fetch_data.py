import os
import pandas as pd
import yfinance as yf


def fetch_crypto_data():
    """
    Fetch historical daily close prices for a fixed universe of major cryptocurrencies
    using Yahoo Finance, then save to data/crypto_prices.csv.
    """

    # Fixed universe of major non-stablecoin crypto assets
    tickers = [
        "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
        "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "LTC-USD",
        "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD", "UNI-USD",
        "XMR-USD", "ETC-USD", "BCH-USD", "XLM-USD", "ALGO-USD"
    ]

    start_date = "2021-01-01"
    end_date = "2024-01-01"

    print(f"Downloading data for {len(tickers)} cryptocurrencies from {start_date} to {end_date}...")

    try:
        raw = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            threads=True
        )
    except Exception as e:
        raise RuntimeError(f"yfinance download failed: {e}")

    if raw.empty:
        raise ValueError("No data returned by yfinance.")

    # Prefer Adj Close if available, otherwise Close
    if isinstance(raw.columns, pd.MultiIndex):
        top_level = raw.columns.get_level_values(0)
        if "Adj Close" in top_level:
            df = raw["Adj Close"].copy()
        elif "Close" in top_level:
            df = raw["Close"].copy()
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' found in downloaded data.")
    else:
        # Fallback for single-ticker or unusual output format
        if "Adj Close" in raw.columns:
            df = raw[["Adj Close"]].copy()
        elif "Close" in raw.columns:
            df = raw[["Close"]].copy()
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' found in downloaded data.")

    # Clean column names
    df.columns = [str(col).replace("-USD", "") for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_index()

    # Basic cleaning
    df = df.ffill()

    # Keep only coins with enough coverage
    min_non_missing = int(len(df) * 0.8)
    df = df.dropna(axis=1, thresh=min_non_missing)

    # Remove any remaining incomplete rows
    df = df.dropna()

    os.makedirs("data", exist_ok=True)
    output_path = "data/crypto_prices.csv"
    df.to_csv(output_path)

    print(f"\nData successfully downloaded and saved to {output_path}")
    print(f"Final dataset shape: {df.shape} (Rows: Dates, Columns: Coins)")
    print(f"Coins retained: {list(df.columns)}")

    return df


if __name__ == "__main__":
    fetch_crypto_data()