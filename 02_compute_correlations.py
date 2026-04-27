import pandas as pd
import numpy as np
import os


def compute_returns_and_correlation(window_size=90, label="base"):
    """
    Load the price data, compute log returns, and calculate rolling correlation matrices.
    """

    input_path = 'data/crypto_prices.csv'
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Please run 01_fetch_data.py first! Cannot find {input_path}")

    df_prices = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df_prices = df_prices.sort_index()
    print(f"Loaded prices shape: {df_prices.shape}")

    df_returns = np.log(df_prices) - np.log(df_prices.shift(1))
    df_returns = df_returns.dropna()
    print(f"Computed returns shape: {df_returns.shape}")

    print(f"Computing rolling correlation matrices with a {window_size}-day window...")

    rolling_corr = df_returns.rolling(window=window_size).corr()
    rolling_corr = rolling_corr.dropna(how='all')

    os.makedirs('data/processed', exist_ok=True)

    returns_path = f'data/processed/crypto_returns_{label}.csv'
    corr_path = f'data/processed/rolling_correlation_{label}.pkl'

    df_returns.to_csv(returns_path)
    rolling_corr.to_pickle(corr_path)

    print(f"Saved returns to {returns_path}")
    print(f"Saved rolling correlations to {corr_path}")

    return df_returns, rolling_corr


if __name__ == "__main__":
    returns, corr_matrices = compute_returns_and_correlation(window_size=90, label="base")

    last_date = corr_matrices.index.get_level_values(0).max()
    print(f"\nCorrelation matrix snapshot on {last_date.date()}:")
    print(corr_matrices.loc[last_date].iloc[:5, :5])