import pandas as pd
import numpy as np
import networkx as nx
import os

def compute_contagion_indicators():
    """
    Load the rolling correlation matrices and compute 5 systemic risk (contagion) 
    indicators for each day.
    """
    
    # 1. Load the rolling correlation matrices saved from Step 2
    input_path = 'data/processed/rolling_correlation.pkl'
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cannot find {input_path}. Please run 02_compute_correlation.py first!")
    
    print("Loading rolling correlation matrices...")
    rolling_corr = pd.read_pickle(input_path)
    
    # Get all unique dates from the MultiIndex (Level 0 is Date)
    dates = rolling_corr.index.get_level_values(0).unique()
    
    # Dictionary to store our daily indicator results
    indicators = {
        'Date': [],
        'Avg_Correlation': [],
        'Upper_Tail_90th': [],
        'Fraction_Above_06': [],
        'Effective_Factors': [],
        'MST_Length': []
    }
    
    print(f"Computing indicators for {len(dates)} trading days...")
    
    # 2. Loop through each day to calculate the 5 metrics
    for date in dates:
        # Extract the correlation matrix for this specific day
        # .values converts the pandas DataFrame to a numpy array for faster math
        corr_matrix = rolling_corr.loc[date].values
        
        # We only look at the upper triangle (off-diagonal) to avoid duplicates 
        # and ignore the self-correlation of 1 on the diagonal.
        # np.triu_indices_from gets the indices for the upper triangle (k=1 excludes diagonal)
        upper_tri_idx = np.triu_indices_from(corr_matrix, k=1)
        pairwise_corrs = corr_matrix[upper_tri_idx]
        
        # Check if the matrix is valid (sometimes missing data causes NaNs)
        if np.isnan(pairwise_corrs).all():
            continue
            
        # Indicator 1: Average pairwise correlation
        avg_corr = np.nanmean(pairwise_corrs)
        
        # Indicator 2: Upper-tail correlation (90th percentile)
        upper_tail = np.nanpercentile(pairwise_corrs, 90)
        
        # Indicator 3: Fraction of highly correlated pairs (Threshold = 0.6)
        total_pairs = len(pairwise_corrs)
        highly_correlated = np.sum(pairwise_corrs > 0.6)
        fraction_above_06 = highly_correlated / total_pairs if total_pairs > 0 else 0
        
        # Indicator 4: Effective number of factors (Eigenvalue concentration)
        # Replace NaNs with 0 to allow eigenvalue computation
        clean_matrix = np.nan_to_num(corr_matrix)
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(clean_matrix)
        # Keep only positive eigenvalues to avoid complex numbers due to rounding errors
        eigenvalues = np.maximum(eigenvalues, 0)
        sum_eigen = np.sum(eigenvalues)
        if sum_eigen > 0:
            p_k = eigenvalues / sum_eigen
            n_eff = 1.0 / np.sum(p_k**2)
        else:
            n_eff = np.nan
            
        # Indicator 5: Minimum Spanning Tree (MST) total length
        # Convert correlation to distance: d = sqrt(2 * (1 - rho))
        # Clip to avoid negative values under sqrt due to float precision
        distance_matrix = np.sqrt(np.clip(2 * (1 - corr_matrix), 0, None))
        
        # Build a graph using NetworkX
        G = nx.from_numpy_array(distance_matrix)
        # Compute the Minimum Spanning Tree
        mst = nx.minimum_spanning_tree(G)
        # Sum the weights (distances) of all edges in the MST
        mst_length = sum(data['weight'] for u, v, data in mst.edges(data=True))
        
        # Append results for this date
        indicators['Date'].append(date)
        indicators['Avg_Correlation'].append(avg_corr)
        indicators['Upper_Tail_90th'].append(upper_tail)
        indicators['Fraction_Above_06'].append(fraction_above_06)
        indicators['Effective_Factors'].append(n_eff)
        indicators['MST_Length'].append(mst_length)

    # 3. Save the results into a clean DataFrame
    df_indicators = pd.DataFrame(indicators)
    df_indicators.set_index('Date', inplace=True)
    
    # Save to CSV for easy plotting later
    output_path = 'data/processed/contagion_indicators.csv'
    df_indicators.to_csv(output_path)
    
    print("\nCalculations complete!")
    print(f"Indicators saved to {output_path}")
    print("\nPreview of the latest 5 days:")
    print(df_indicators.tail())
    
    return df_indicators

if __name__ == "__main__":
    compute_contagion_indicators()