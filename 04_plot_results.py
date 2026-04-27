import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_contagion_indicators():
    """
    Load the computed contagion indicators and generate a 5-panel subplot 
    dashboard to visualize how systemic risk evolves over time.
    """
    
    # 1. Load the indicators data
    input_path = 'data/processed/contagion_indicators.csv'
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cannot find {input_path}. Please run 03_compute_indicators.py first!")
    
    print("Loading contagion indicators for visualization...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    # 2. Create an output directory for the figures
    os.makedirs('figures', exist_ok=True)
    
    # 3. Set up the matplotlib figure with 5 subplots (one for each indicator)
    # figsize=(12, 16) makes it tall enough to fit all 5 charts clearly
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 16), sharex=True)
    fig.suptitle('Dynamic Contagion Indicators in Cryptocurrency Market', fontsize=16, y=0.92)
    
    # Indicator 1: Average Correlation
    axes[0].plot(df.index, df['Avg_Correlation'], color='blue', linewidth=1.5)
    axes[0].set_title('1. Average Pairwise Correlation (Higher = More Contagion)')
    axes[0].set_ylabel('Correlation')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Indicator 2: Upper-Tail Correlation (90th percentile)
    axes[1].plot(df.index, df['Upper_Tail_90th'], color='red', linewidth=1.5)
    axes[1].set_title('2. Upper-Tail Correlation (90th Percentile)')
    axes[1].set_ylabel('Correlation')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # Indicator 3: Fraction Above 0.6
    axes[2].plot(df.index, df['Fraction_Above_06'], color='purple', linewidth=1.5)
    axes[2].set_title('3. Fraction of Highly Correlated Pairs (> 0.6)')
    axes[2].set_ylabel('Fraction')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    # Indicator 4: Effective Number of Factors (N_eff)
    axes[3].plot(df.index, df['Effective_Factors'], color='orange', linewidth=1.5)
    axes[3].set_title('4. Effective Number of Factors (Lower = More Centralized / Contagion)')
    axes[3].set_ylabel('Number of Factors')
    axes[3].grid(True, linestyle='--', alpha=0.6)
    
    # Indicator 5: Minimum Spanning Tree (MST) Length
    axes[4].plot(df.index, df['MST_Length'], color='green', linewidth=1.5)
    axes[4].set_title('5. Minimum Spanning Tree Total Length (Lower = Tighter Network)')
    axes[4].set_ylabel('Distance')
    axes[4].set_xlabel('Date')
    axes[4].grid(True, linestyle='--', alpha=0.6)
    
    # 4. Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.9]) # Leave space for the main title
    output_path = 'figures/contagion_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight') # High resolution for reports
    
    print(f"\nSuccess! Dashboard image saved to {output_path}")
    
if __name__ == "__main__":
    plot_contagion_indicators()