import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_mst_from_returns(df_returns):
    corr = df_returns.corr().values
    clean_corr = np.nan_to_num(corr, nan=0.0)
    distance_matrix = np.sqrt(np.clip(2 * (1 - clean_corr), 0, None))
    G = nx.from_numpy_array(distance_matrix)
    mst = nx.minimum_spanning_tree(G)
    return mst


def plot_mst_graphs():
    """
    Plot MST graphs for calm and stress samples.
    """
    prices_path = "data/crypto_prices.csv"
    event_path = "data/processed/event_stress_windows.csv"
    data_driven_path = "data/processed/data_driven_stress_windows.csv"

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

    # Use a single snapshot correlation for each regime
    calm_mst = build_mst_from_returns(calm_returns)
    stress_mst = build_mst_from_returns(stress_returns)

    labels = list(df_returns.columns)

    os.makedirs("figures", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    pos1 = nx.spring_layout(calm_mst, seed=42)
    nx.draw_networkx(
        calm_mst,
        pos=pos1,
        labels={i: labels[i] for i in range(len(labels))},
        node_size=400,
        font_size=7,
        ax=axes[0],
        edge_color="gray"
    )
    axes[0].set_title("Calm Period MST")

    pos2 = nx.spring_layout(stress_mst, seed=42)
    nx.draw_networkx(
        stress_mst,
        pos=pos2,
        labels={i: labels[i] for i in range(len(labels))},
        node_size=400,
        font_size=7,
        ax=axes[1],
        edge_color="red"
    )
    axes[1].set_title("Stress Period MST")

    output_path = "figures/calm_vs_stress_mst.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved MST graphs to {output_path}")


if __name__ == "__main__":
    plot_mst_graphs()