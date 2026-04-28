import os
import pandas as pd
import numpy as np
import networkx as nx


def compute_contagion_indicators_from_prices(df_prices, window_size=90):
    """
    Compute contagion indicators from a price dataframe.
    """
    df_prices = df_prices.sort_index()
    df_returns = np.log(df_prices).diff().dropna()
    rolling_corr = df_returns.rolling(window=window_size).corr().dropna(how="all")

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
    return df_indicators


def universe_sensitivity_check():
    """
    Compare contagion indicators under different coin universes.
    """
    input_path = "data/crypto_prices.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cannot find {input_path}. Run 01_fetch_data.py first.")

    df_prices = pd.read_csv(input_path, index_col=0, parse_dates=True).sort_index()

    all_coins = list(df_prices.columns)

    # Main universe: all available coins
    main_universe = all_coins

    # Smaller universe: top 20 by default, if available
    smaller_universe = all_coins[:20] if len(all_coins) >= 20 else all_coins

    configs = [
        {"label": "main", "coins": main_universe},
        {"label": "smaller", "coins": smaller_universe},
    ]

    os.makedirs("data/processed", exist_ok=True)

    rows = []
    for cfg in configs:
        print(f"\n=== Running universe sensitivity: {cfg['label']} ({len(cfg['coins'])} coins) ===")
        sub_prices = df_prices[cfg["coins"]].copy()
        indicators = compute_contagion_indicators_from_prices(sub_prices, window_size=90)
        out_path = f"data/processed/contagion_indicators_universe_{cfg['label']}.csv"
        indicators.to_csv(out_path)
        print(f"Saved indicators to {out_path}")

        summary_row = {
            "Universe": cfg["label"],
            "Num_Coins": len(cfg["coins"]),
            "Avg_Correlation_Mean": indicators["Avg_Correlation"].mean(),
            "Upper_Tail_90th_Mean": indicators["Upper_Tail_90th"].mean(),
            "Fraction_Above_06_Mean": indicators["Fraction_Above_06"].mean(),
            "Effective_Factors_Mean": indicators["Effective_Factors"].mean(),
            "MST_Length_Mean": indicators["MST_Length"].mean(),
        }
        rows.append(summary_row)

    summary = pd.DataFrame(rows)
    summary_path = "data/processed/universe_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved universe sensitivity summary to {summary_path}")
    print(summary)

    return summary


if __name__ == "__main__":
    universe_sensitivity_check()