import os
import numpy as np
import pandas as pd


def max_drawdown(series):
    """
    Compute maximum drawdown of a return series.
    Input: daily returns
    Output: maximum drawdown value (negative number)
    """
    cumulative = (1 + series).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return drawdown.min()


def portfolio_risk_linkage(
    indicators_path="data/processed/contagion_indicators.csv",
    returns_path="data/processed/crypto_returns.csv",
    output_path="data/processed/portfolio_risk_linkage.csv",
    contagion_quantile=0.90,
    rolling_vol_window=30
):
    """
    Compare portfolio risk in high-contagion vs low-contagion regimes.

    Steps:
    1. Load contagion indicators and asset returns
    2. Build an equal-weight crypto portfolio
    3. Define high-contagion regime using the chosen indicator quantile
    4. Compare portfolio volatility and drawdown in high vs low contagion periods
    """

    if not os.path.exists(indicators_path):
        raise FileNotFoundError(f"Cannot find {indicators_path}. Run 04_compute_indicators.py first.")
    if not os.path.exists(returns_path):
        raise FileNotFoundError(f"Cannot find {returns_path}. Run 02_compute_correlation.py first.")

    # Load data
    df_ind = pd.read_csv(indicators_path, index_col=0, parse_dates=True).sort_index()
    df_ret = pd.read_csv(returns_path, index_col=0, parse_dates=True).sort_index()

    # Align dates
    common_dates = df_ind.index.intersection(df_ret.index)
    df_ind = df_ind.loc[common_dates].copy()
    df_ret = df_ret.loc[common_dates].copy()

    if df_ind.empty or df_ret.empty:
        raise ValueError("No overlapping dates between indicators and returns.")

    # Equal-weight portfolio return
    portfolio_return = df_ret.mean(axis=1)
    portfolio_cum = (1 + portfolio_return).cumprod()

    # Rolling portfolio volatility
    portfolio_vol = portfolio_return.rolling(window=rolling_vol_window).std()

    # Use Avg_Correlation to define high-contagion periods
    contagion_series = df_ind["Avg_Correlation"].copy()
    threshold = contagion_series.quantile(contagion_quantile)
    high_contagion = (contagion_series >= threshold).astype(int)

    # Summary stats
    high_mask = high_contagion == 1
    low_mask = high_contagion == 0

    high_returns = portfolio_return.loc[high_mask]
    low_returns = portfolio_return.loc[low_mask]

    high_vol = portfolio_vol.loc[high_mask].dropna()
    low_vol = portfolio_vol.loc[low_mask].dropna()

    # Drawdown by regime
    high_drawdown = max_drawdown(high_returns) if len(high_returns) > 0 else np.nan
    low_drawdown = max_drawdown(low_returns) if len(low_returns) > 0 else np.nan

    summary = pd.DataFrame([
        {
            "Regime": "High_Contagion",
            "Count": len(high_returns),
            "Mean_Return": high_returns.mean(),
            "Std_Return": high_returns.std(),
            "Mean_Rolling_Vol": high_vol.mean() if len(high_vol) > 0 else np.nan,
            "Max_Drawdown": high_drawdown
        },
        {
            "Regime": "Low_Contagion",
            "Count": len(low_returns),
            "Mean_Return": low_returns.mean(),
            "Std_Return": low_returns.std(),
            "Mean_Rolling_Vol": low_vol.mean() if len(low_vol) > 0 else np.nan,
            "Max_Drawdown": low_drawdown
        }
    ])

    # Save a merged daily dataset for plotting or inspection
    merged = pd.DataFrame({
        "Portfolio_Return": portfolio_return,
        "Portfolio_Cumulative": portfolio_cum,
        "Portfolio_Rolling_Vol": portfolio_vol,
        "Avg_Correlation": contagion_series,
        "High_Contagion": high_contagion
    })

    os.makedirs("data/processed", exist_ok=True)
    summary.to_csv(output_path, index=False)
    merged.to_csv("data/processed/portfolio_contagion_merged.csv")

    print("\nPortfolio-risk linkage complete.")
    print(summary)

    return summary, merged


if __name__ == "__main__":
    portfolio_risk_linkage()