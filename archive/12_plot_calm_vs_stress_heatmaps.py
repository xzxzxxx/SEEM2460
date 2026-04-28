import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_calm_vs_stress_heatmaps():
    """
    Plot average correlation heatmaps for calm vs stress periods.
    """
    indicators_path = "data/processed/contagion_indicators.csv"
    event_path = "data/processed/event_stress_windows.csv"
    data_driven_path = "data/processed/data_driven_stress_windows.csv"
    prices_path = "data/crypto_prices.csv"

    if not os.path.exists(indicators_path):
        raise FileNotFoundError(f"Cannot find {indicators_path}. Run 05_compute_indicators.py first.")
    if not os.path.exists(prices_path):
        raise FileNotFoundError(f"Cannot find {prices_path}. Run 01_fetch_data.py first.")

    df_prices = pd.read_csv(prices_path, index_col=0, parse_dates=True).sort_index()
    df_returns = np.log(df_prices).diff().dropna()

    event_dates = set()
    data_dates = set()

    if os.path.exists(event_path):
        event_df = pd.read_csv(event_path, parse_dates=["Date"])
        event_dates = set(event_df["Date"].dt.normalize().tolist())

    if os.path.exists(data_driven_path):
        data_df = pd.read_csv(data_driven_path, parse_dates=["Date"])
        data_dates = set(data_df.loc[data_df["Stress_Label"] == 1, "Date"].dt.normalize().tolist())

    common_index = df_returns.index.normalize()

    calm_mask = (~common_index.isin(event_dates)) & (~common_index.isin(data_dates))
    stress_mask = common_index.isin(event_dates) | common_index.isin(data_dates)

    calm_returns = df_returns.loc[calm_mask]
    stress_returns = df_returns.loc[stress_mask]

    calm_corr = calm_returns.corr()
    stress_corr = stress_returns.corr()

    os.makedirs("figures", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    sns.heatmap(calm_corr, ax=axes[0], cmap="coolwarm", vmin=-1, vmax=1, square=True)
    axes[0].set_title("Calm Period Correlation Heatmap")

    sns.heatmap(stress_corr, ax=axes[1], cmap="coolwarm", vmin=-1, vmax=1, square=True)
    axes[1].set_title("Stress Period Correlation Heatmap")

    output_path = "figures/calm_vs_stress_heatmaps.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved heatmaps to {output_path}")


if __name__ == "__main__":
    plot_calm_vs_stress_heatmaps()