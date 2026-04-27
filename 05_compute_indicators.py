import pandas as pd
import numpy as np
import networkx as nx
import os


def compute_contagion_indicators():
    """
    Load the rolling correlation matrices and compute 5 systemic risk (contagion)
    indicators for each day.
    """

    input_path = 'data/processed/rolling_correlation.pkl'
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cannot find {input_path}. Please run 02_compute_correlation.py first!")

    print("Loading rolling correlation matrices...")
    rolling_corr = pd.read_pickle(input_path)

    dates = rolling_corr.index.get_level_values(0).unique()

    indicators = {
        'Date': [],
        'Avg_Correlation': [],
        'Upper_Tail_90th': [],
        'Fraction_Above_06': [],
        'Effective_Factors': [],
        'MST_Length': []
    }

    print(f"Computing indicators for {len(dates)} trading days...")

    for date in dates:
        corr_matrix = rolling_corr.loc[date].values

        # Upper triangle only, excluding diagonal
        upper_tri_idx = np.triu_indices_from(corr_matrix, k=1)
        pairwise_corrs = corr_matrix[upper_tri_idx]

        # Skip if all values are missing
        if np.isnan(pairwise_corrs).all():
            continue

        # 1. Average pairwise correlation
        avg_corr = np.nanmean(pairwise_corrs)

        # 2. Upper-tail correlation (90th percentile)
        upper_tail = np.nanpercentile(pairwise_corrs, 90)

        # 3. Fraction of highly correlated pairs
        total_pairs = np.sum(~np.isnan(pairwise_corrs))
        highly_correlated = np.sum(pairwise_corrs > 0.6)
        fraction_above_06 = highly_correlated / total_pairs if total_pairs > 0 else np.nan

        # 4. Effective number of factors
        clean_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        eigenvalues = np.linalg.eigvalsh(clean_matrix)
        eigenvalues = np.maximum(eigenvalues, 0)

        sum_eigen = np.sum(eigenvalues)
        if sum_eigen > 0:
            p_k = eigenvalues / sum_eigen
            n_eff = 1.0 / np.sum(p_k**2)
        else:
            n_eff = np.nan

        # 5. Minimum Spanning Tree total length
        distance_matrix = np.sqrt(np.clip(2 * (1 - clean_matrix), 0, None))
        G = nx.from_numpy_array(distance_matrix)
        mst = nx.minimum_spanning_tree(G)
        mst_length = sum(data['weight'] for _, _, data in mst.edges(data=True))

        indicators['Date'].append(date)
        indicators['Avg_Correlation'].append(avg_corr)
        indicators['Upper_Tail_90th'].append(upper_tail)
        indicators['Fraction_Above_06'].append(fraction_above_06)
        indicators['Effective_Factors'].append(n_eff)
        indicators['MST_Length'].append(mst_length)

    df_indicators = pd.DataFrame(indicators)
    df_indicators.set_index('Date', inplace=True)

    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/contagion_indicators.csv'
    df_indicators.to_csv(output_path)

    print("\nCalculations complete!")
    print(f"Indicators saved to {output_path}")
    print("\nPreview of the latest 5 days:")
    print(df_indicators.tail())

    return df_indicators


if __name__ == "__main__":
    compute_contagion_indicators()