import os
import pandas as pd
import numpy as np
import networkx as nx


def compute_rolling_correlation(df_returns, window_size=90, method="pearson"):
    """
    Compute rolling correlation matrices using Pearson or Spearman.
    For Spearman, rank data within each window first.
    """
    dates = df_returns.index
    assets = df_returns.columns

    corr_list = []
    corr_index = []

    for end_idx in range(window_size - 1, len(df_returns)):
        window = df_returns.iloc[end_idx - window_size + 1:end_idx + 1]

        if method == "spearman":
            window = window.rank(axis=0)

        corr_matrix = window.corr(method="pearson")
        corr_list.append(corr_matrix)
        corr_index.append(dates[end_idx])

    # Convert list of DataFrames into MultiIndex DataFrame
    rolling_corr = pd.concat(corr_list, keys=corr_index, names=["Date", "Asset"])
    return rolling_corr


def compute_contagion_indicators_from_corr(rolling_corr, output_label):
    """
    Compute the 5 contagion indicators from rolling correlation matrices.
    """
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


def run_spearman_robustness():
    """
    Run Pearson and Spearman rolling-correlation robustness checks.
    """
    input_path = "data/crypto_prices.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cannot find {input_path}. Run 01_fetch_data.py first.")

    df_prices = pd.read_csv(input_path, index_col=0, parse_dates=True).sort_index()
    df_returns = np.log(df_prices).diff().dropna()

    configs = [
        {"method": "pearson", "label": "pearson"},
        {"method": "spearman", "label": "spearman"},
    ]

    for cfg in configs:
        print(f"\n=== Running {cfg['label']} correlation robustness ===")
        rolling_corr = compute_rolling_correlation(
            df_returns=df_returns,
            window_size=90,
            method=cfg["method"]
        )

        os.makedirs("data/processed", exist_ok=True)
        corr_path = f"data/processed/rolling_correlation_{cfg['label']}.pkl"
        rolling_corr.to_pickle(corr_path)

        compute_contagion_indicators_from_corr(
            rolling_corr=rolling_corr,
            output_label=cfg["label"]
        )

    print("\nSpearman robustness run complete.")


if __name__ == "__main__":
    run_spearman_robustness()