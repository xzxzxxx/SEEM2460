import os
import pandas as pd
import numpy as np
import networkx as nx


def compute_returns_and_correlation(window_size=90, label="base"):
    """
    Load price data, compute log returns, and calculate rolling correlation matrices.
    Saves outputs with a label so multiple window lengths can coexist.
    """
    input_path = "data/crypto_prices.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cannot find {input_path}. Run 01_fetch_data.py first.")

    df_prices = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df_prices = df_prices.sort_index()

    df_returns = np.log(df_prices).diff().dropna()

    rolling_corr = df_returns.rolling(window=window_size).corr()
    rolling_corr = rolling_corr.dropna(how="all")

    os.makedirs("data/processed", exist_ok=True)
    returns_path = f"data/processed/crypto_returns_{label}.csv"
    corr_path = f"data/processed/rolling_correlation_{label}.pkl"

    df_returns.to_csv(returns_path)
    rolling_corr.to_pickle(corr_path)

    print(f"Saved returns to {returns_path}")
    print(f"Saved rolling correlations to {corr_path}")

    return df_returns, rolling_corr


def compute_contagion_indicators(corr_path, output_label):
    """
    Load rolling correlation matrices and compute contagion indicators.
    """
    if not os.path.exists(corr_path):
        raise FileNotFoundError(f"Cannot find {corr_path}")

    rolling_corr = pd.read_pickle(corr_path)
    dates = rolling_corr.index.get_level_values(0).unique()

    indicators = {
        "Date": [],
        "Avg_Correlation": [],
        "Upper_Tail_90th": [],
        "Fraction_Above_06": [],
        "Effective_Factors": [],
        "MST_Length": [],
    }

    for date in dates:
        corr_matrix = rolling_corr.loc[date].values

        if corr_matrix.ndim != 2 or corr_matrix.shape[0] != corr_matrix.shape[1]:
            continue

        upper_tri_idx = np.triu_indices_from(corr_matrix, k=1)
        pairwise_corrs = corr_matrix[upper_tri_idx]

        if np.isnan(pairwise_corrs).all():
            continue

        avg_corr = np.nanmean(pairwise_corrs)
        upper_tail = np.nanpercentile(pairwise_corrs, 90)

        total_pairs = np.sum(~np.isnan(pairwise_corrs))
        highly_correlated = np.sum(pairwise_corrs > 0.6)
        fraction_above_06 = highly_correlated / total_pairs if total_pairs > 0 else np.nan

        clean_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        eigenvalues = np.linalg.eigvalsh(clean_matrix)
        eigenvalues = np.maximum(eigenvalues, 0)

        sum_eigen = np.sum(eigenvalues)
        if sum_eigen > 0:
            p_k = eigenvalues / sum_eigen
            n_eff = 1.0 / np.sum(p_k ** 2)
        else:
            n_eff = np.nan

        distance_matrix = np.sqrt(np.clip(2 * (1 - clean_matrix), 0, None))
        G = nx.from_numpy_array(distance_matrix)
        mst = nx.minimum_spanning_tree(G)
        mst_length = sum(data["weight"] for _, _, data in mst.edges(data=True))

        indicators["Date"].append(date)
        indicators["Avg_Correlation"].append(avg_corr)
        indicators["Upper_Tail_90th"].append(upper_tail)
        indicators["Fraction_Above_06"].append(fraction_above_06)
        indicators["Effective_Factors"].append(n_eff)
        indicators["MST_Length"].append(mst_length)

    df_indicators = pd.DataFrame(indicators).set_index("Date")

    os.makedirs("data/processed", exist_ok=True)
    out_path = f"data/processed/contagion_indicators_{output_label}.csv"
    df_indicators.to_csv(out_path)

    print(f"Saved indicators to {out_path}")
    return df_indicators


def run_rolling_window_robustness():
    """
    Run the full pipeline for short and long rolling windows.
    """
    configs = [
        {"window_size": 60, "label": "short"},
        {"window_size": 90, "label": "long"},
    ]

    for cfg in configs:
        print(f"\n=== Running {cfg['label']} window ({cfg['window_size']} days) ===")
        _, _ = compute_returns_and_correlation(window_size=cfg["window_size"], label=cfg["label"])
        corr_path = f"data/processed/rolling_correlation_{cfg['label']}.pkl"
        _ = compute_contagion_indicators(corr_path=corr_path, output_label=cfg["label"])

    print("\nRolling-window robustness run complete.")


if __name__ == "__main__":
    run_rolling_window_robustness()