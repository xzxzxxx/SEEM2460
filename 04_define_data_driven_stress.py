import os
import pandas as pd
import numpy as np


def define_data_driven_stress_windows(
    vol_window=30,
    stress_percentile=0.90
):
    """
    Define data-driven stress windows using rolling market volatility.
    Stress = top (1 - stress_percentile) volatility periods.
    """

    input_path = "data/processed/crypto_returns.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cannot find {input_path}. Run 02_compute_correlation.py first.")

    df_returns = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df_returns = df_returns.sort_index()

    # Equal-weight market return proxy
    market_return = df_returns.mean(axis=1)

    # Rolling volatility
    rolling_vol = market_return.rolling(window=vol_window).std()

    # Threshold for stress
    threshold = rolling_vol.quantile(stress_percentile)

    df_stress = pd.DataFrame({
        "Date": rolling_vol.index,
        "Market_Return": market_return.values,
        "Rolling_Volatility": rolling_vol.values,
        "Stress_Label": (rolling_vol >= threshold).astype(int)
    })

    # Remove initial NaN rows from rolling window
    df_stress = df_stress.dropna()

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/data_driven_stress_windows.csv"
    df_stress.to_csv(output_path, index=False)

    print(f"Saved data-driven stress windows to {output_path}")
    print(f"Volatility window: {vol_window} days")
    print(f"Stress threshold: top {(1 - stress_percentile) * 100:.0f}% volatility periods")
    print(f"Number of stressed days: {df_stress['Stress_Label'].sum()}")

    return df_stress


if __name__ == "__main__":
    define_data_driven_stress_windows()