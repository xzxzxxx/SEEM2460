import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def plot_dashboard():
    ind_path = "data/processed/contagion_indicators.csv"
    if not os.path.exists(ind_path):
        raise FileNotFoundError("Run data_pipeline.py first.")

    df = pd.read_csv(ind_path, parse_dates=["Date"]).sort_values("Date")

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    axes[0].plot(df["Date"], df["Avg_Correlation"], label="Avg Correlation", color="tab:blue")
    axes[0].set_title("Average Pairwise Correlation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["Date"], df["Upper_Tail_90th"], label="90th Percentile Pairwise Correlation", color="tab:orange")
    axes[1].set_title("Upper Tail Correlation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["Date"], df["Fraction_Above_06"], label="Fraction Above 0.6", color="tab:red")
    axes[2].set_title("Fraction of Pairwise Correlations Above 0.6")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(df["Date"], df["Effective_Factors"], label="Effective Factors", color="tab:green")
    axes[3].set_title("Effective Number of Factors")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(df["Date"], df["MST_Length"], label="MST Length", color="tab:purple")
    axes[4].set_title("Minimum Spanning Tree Length")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "figures/contagion_dashboard.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved dashboard to {out_path}")


def plot_indicator_overlays():
    base_path = "data/processed/contagion_indicators.csv"
    roll_path = "data/processed/contagion_indicators_rolling_window.csv"
    if not os.path.exists(base_path) or not os.path.exists(roll_path):
        raise FileNotFoundError("Run data_pipeline.py and robustness.py first.")

    base = pd.read_csv(base_path, parse_dates=["Date"]).sort_values("Date")
    roll = pd.read_csv(roll_path, parse_dates=["Date"]).sort_values(["Window", "Date"])

    metrics = [
        "Avg_Correlation",
        "Upper_Tail_90th",
        "Fraction_Above_06",
        "Effective_Factors",
        "MST_Length",
    ]

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    colors = {60: "tab:blue", 90: "tab:orange", 120: "tab:green"}

    for i, metric in enumerate(metrics):
        axes[i].plot(base["Date"], base[metric], color="black", linewidth=2, label="Main 90D rolling")
        for w in sorted(roll["Window"].unique()):
            sub = roll.loc[roll["Window"] == w]
            axes[i].plot(sub["Date"], sub[metric], alpha=0.7, color=colors.get(w, None), label=f"{w}D window")
        axes[i].set_title(metric)
        axes[i].legend(ncol=2)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "figures/indicator_overlays.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved indicator overlays to {out_path}")


def plot_heatmaps():
    prices_path = "data/crypto_prices.csv"
    event_path = "data/processed/event_stress_windows.csv"
    data_driven_path = "data/processed/data_driven_stress_windows.csv"

    if not os.path.exists(prices_path):
        raise FileNotFoundError("Run data_pipeline.py first.")

    df_prices = pd.read_csv(prices_path, index_col=0, parse_dates=True).sort_index()
    df_returns = np.log(df_prices).diff().dropna()

    event_dates = set()
    data_dates = set()

    if os.path.exists(event_path):
        event_df = pd.read_csv(event_path, parse_dates=["Date"])
        event_dates = set(event_df["Date"].dt.normalize().tolist())

    if os.path.exists(data_driven_path):
        data_df = pd.read_csv(data_driven_path, parse_dates=["Date"])
        data_dates = set(
            data_df.loc[data_df["Stress_Label"] == 1, "Date"].dt.normalize().tolist()
        )

    idx = df_returns.index.normalize()
    calm_mask = (~idx.isin(event_dates)) & (~idx.isin(data_dates))
    stress_mask = idx.isin(event_dates) | idx.isin(data_dates)

    calm_corr = df_returns.loc[calm_mask].corr()
    stress_corr = df_returns.loc[stress_mask].corr()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    sns.heatmap(calm_corr, ax=axes[0], cmap="coolwarm", vmin=-1, vmax=1, square=True)
    axes[0].set_title("Calm Period Correlation Heatmap")

    sns.heatmap(stress_corr, ax=axes[1], cmap="coolwarm", vmin=-1, vmax=1, square=True)
    axes[1].set_title("Stress Period Correlation Heatmap")

    out_path = "figures/calm_vs_stress_heatmaps.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmaps to {out_path}")


def plot_correlation_distributions():
    prices_path = "data/crypto_prices.csv"
    event_path = "data/processed/event_stress_windows.csv"
    data_driven_path = "data/processed/data_driven_stress_windows.csv"

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

    idx = df_returns.index.normalize()
    calm_mask = (~idx.isin(event_dates)) & (~idx.isin(data_dates))
    stress_mask = idx.isin(event_dates) | idx.isin(data_dates)

    calm_corr = df_returns.loc[calm_mask].corr().values
    stress_corr = df_returns.loc[stress_mask].corr().values

    def upper_tri_values(mat):
        iu = np.triu_indices_from(mat, k=1)
        vals = mat[iu]
        return vals[~np.isnan(vals)]

    calm_vals = upper_tri_values(calm_corr)
    stress_vals = upper_tri_values(stress_corr)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(calm_vals, bins=30, color="tab:blue", stat="density", label="Calm", kde=True, alpha=0.45)
    sns.histplot(stress_vals, bins=30, color="tab:red", stat="density", label="Stress", kde=True, alpha=0.45)
    ax.set_title("Correlation Distributions: Calm vs Stress")
    ax.set_xlabel("Pairwise Correlation")
    ax.legend()

    out_path = "figures/correlation_distributions.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation distributions to {out_path}")


def plot_mst_graphs():
    prices_path = "data/crypto_prices.csv"
    event_path = "data/processed/event_stress_windows.csv"
    data_driven_path = "data/processed/data_driven_stress_windows.csv"

    if not os.path.exists(prices_path):
        raise FileNotFoundError("Run data_pipeline.py first.")

    df_prices = pd.read_csv(prices_path, index_col=0, parse_dates=True).sort_index()
    df_returns = np.log(df_prices).diff().dropna()

    event_dates = set()
    data_dates = set()

    if os.path.exists(event_path):
        event_df = pd.read_csv(event_path, parse_dates=["Date"])
        event_dates = set(event_df["Date"].dt.normalize().tolist())

    if os.path.exists(data_driven_path):
        data_df = pd.read_csv(data_driven_path, parse_dates=["Date"])
        data_dates = set(
            data_df.loc[data_df["Stress_Label"] == 1, "Date"].dt.normalize().tolist()
        )

    idx = df_returns.index.normalize()
    calm_mask = (~idx.isin(event_dates)) & (~idx.isin(data_dates))
    stress_mask = idx.isin(event_dates) | idx.isin(data_dates)

    calm_corr = df_returns.loc[calm_mask].corr().values
    stress_corr = df_returns.loc[stress_mask].corr().values

    def corr_to_mst(corr_matrix):
        clean = np.nan_to_num(corr_matrix, nan=0.0)
        dist = np.sqrt(np.clip(2 * (1 - clean), 0, None))
        G = nx.from_numpy_array(dist)
        return nx.minimum_spanning_tree(G)

    calm_mst = corr_to_mst(calm_corr)
    stress_mst = corr_to_mst(stress_corr)

    labels = list(df_returns.columns)

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

    out_path = "figures/calm_vs_stress_mst.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved MST graphs to {out_path}")


def plot_portfolio_risk_comparison():
    path = "data/processed/portfolio_risk_comparison.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Run robustness.py first.")

    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

    sns.barplot(data=df, x="Regime", y="Mean_Return", ax=axes[0])
    axes[0].set_title("Mean Return")

    sns.barplot(data=df, x="Regime", y="Volatility", ax=axes[1])
    axes[1].set_title("Volatility")

    sns.barplot(data=df, x="Regime", y="Max_Drawdown_Proxy", ax=axes[2])
    axes[2].set_title("Drawdown Proxy")

    out_path = "figures/portfolio_risk_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved portfolio risk comparison to {out_path}")


def run_visualizations():
    ensure_dirs()
    plot_dashboard()
    plot_indicator_overlays()
    plot_heatmaps()
    plot_correlation_distributions()
    plot_mst_graphs()
    plot_portfolio_risk_comparison()


if __name__ == "__main__":
    run_visualizations()