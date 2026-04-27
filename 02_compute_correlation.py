import pandas as pd
import numpy as np
import os

def compute_returns_and_correlation(window_size=90):
    """
    Load the price data, compute log returns, and calculate rolling correlation matrices.
    We use a 90-day rolling window as a default for long-term dependence.
    """
    
    # 1. Load the data
    input_path = 'data/crypto_prices.csv'
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Please run 01_fetch_data.py first! Cannot find {input_path}")
    
    # Read CSV and set the first column (Date) as the index
    df_prices = pd.read_csv(input_path, index_col=0, parse_dates=True)
    print(f"Loaded prices shape: {df_prices.shape}")
    
    # 2. Compute Log Returns
    # Formula: r_{t,i} = ln(P_{t,i}) - ln(P_{t-1,i})
    # This is standard practice in quantitative finance
    df_returns = np.log(df_prices) - np.log(df_prices.shift(1))
    
    # Drop the first row since the return is NaN
    df_returns = df_returns.dropna()
    print(f"Computed returns shape: {df_returns.shape}")
    
    # 3. Compute Rolling Correlation Matrices
    # We use pandas rolling().corr() to compute the pairwise correlation over time
    print(f"Computing rolling correlation matrices with a {window_size}-day window...")
    
    # This returns a MultiIndex DataFrame (Date -> Coin -> Coin Correlation)
    rolling_corr = df_returns.rolling(window=window_size).corr()
    
    # Drop the initial NaN periods (the first window_size - 1 days)
    rolling_corr = rolling_corr.dropna(how='all')
    
    # Save returns for future use
    os.makedirs('data/processed', exist_ok=True)
    df_returns.to_csv('data/processed/crypto_returns.csv')
    
    # Save rolling correlation. 
    # Since it's a 3D dataset (Time x Coin x Coin), saving as a pickle is much easier than CSV
    rolling_corr.to_pickle('data/processed/rolling_correlation.pkl')
    
    print("Successfully saved returns and rolling correlations!")
    return df_returns, rolling_corr

if __name__ == "__main__":
    returns, corr_matrices = compute_returns_and_correlation(window_size=90)
    
    # Quick sanity check: Print the correlation matrix for the last available date
    last_date = corr_matrices.index.get_level_values(0).max()
    print(f"\nCorrelation matrix snapshot on {last_date.date()}:")
    print(corr_matrices.loc[last_date].iloc[:5, :5]) # Print top-left 5x5 corner