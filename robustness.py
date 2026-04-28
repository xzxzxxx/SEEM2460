import os
import pandas as pd
import numpy as np
import networkx as nx


def ensure_dirs():
    os.makedirs("data/processed", exist_ok=True)


def compute_indicators_from_returns(df_returns):
    corr = df_returns.corr().values
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        return None

    upper_tri_idx = np.triu_indices_from(corr, k=1)
    pairwise_corrs = corr[upper_tri_idx]
    pairwise_corrs = pairwise_corrs[~np.isnan(pairwise_corrs)]

    if len(pairwise_corrs) == 0:
        return None

    clean_matrix = np.nan_to_num(corr, nan=0.0)

    eigenvalues = np.linalg.eigvalsh(clean_matrix)
    eigenvalues = np.maximum(eigenvalues, 0)
    sum_eigen = np.sum(eigenvalues)
    if sum_eigen > 0:
        p_k = eigenvalues / sum_eigen
        effective_factors = 1.0 / np.sum(p_k ** 2)
    else:
        effective_factors = np.nan

    distance_matrix = np.sqrt(np.clip(2 * (1 - clean_matrix), 0, None))
    G = nx.from_numpy_array(distance_matrix)
    mst = nx.minimum_spanning_tree(G)
    mst_length = sum(data["weight"] for _, _, data in mst.edges(data=True))

    return {
        "Avg_Correlation": np.mean(pairwise_corrs),
        "Upper_Tail_90th": np.percentile(pairwise_corrs, 90),
        "Fraction_Above_06": np.mean(pairwise_corrs > 0.6),
        "Effective_Factors": effective_factors,
        "MST_Length": mst_length,
    }


def run_rolling_window_robustness():
    prices_path = "data/crypto_prices.csv"
    if not os.path.exists(prices_path):
        raise FileNotFoundError("Run data_pipeline.py first.")

    df_prices = pd.read_csv(prices_path, index_col=0, parse_dates=True).sort_index()
    df_returns = np.log(df_prices).diff().dropna()

    windows = [60, 90, 120]
    rows = []

    for w in windows:
        for i in range(w - 1, len(df_returns)):
            sample = df_returns.iloc[i - w + 1:i + 1]
            indicators = compute_indicators_from_returns(sample)
            if indicators is None:
                continue

            rows.append({
                "Date": df_returns.index[i],
                "Window": w,
                **indicators
            })

    result = pd.DataFrame(rows)
    out_path = "data/processed/contagion_indicators_rolling_window.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved rolling window robustness to {out_path}")

    return result


def run_spearman_robustness():
    prices_path = "data/crypto_prices.csv"
    if not os.path.exists(prices_path):
        raise FileNotFoundError("Run data_pipeline.py first.")

    df_prices = pd.read_csv(prices_path, index_col=0, parse_dates=True).sort_index()
    df_returns = np.log(df_prices).diff().dropna()

    window = 90
    rows = []

    for i in range(window - 1, len(df_returns)):
        sample = df_returns.iloc[i - window + 1:i + 1]
        corr = sample.corr(method="spearman").values

        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            continue

        upper_tri_idx = np.triu_indices_from(corr, k=1)
        pairwise_corrs = corr[upper_tri_idx]
        pairwise_corrs = pairwise_corrs[~np.isnan(pairwise_corrs)]

        if len(pairwise_corrs) == 0:
            continue

        rows.append({
            "Date": df_returns.index[i],
            "Avg_Correlation": np.mean(pairwise_corrs),
            "Upper_Tail_90th": np.percentile(pairwise_corrs, 90),
            "Fraction_Above_06": np.mean(pairwise_corrs > 0.6),
        })

    result = pd.DataFrame(rows)
    out_path = "data/processed/contagion_indicators_spearman.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved Spearman robustness to {out_path}")

    return result


def run_universe_sensitivity():
    prices_path = "data/crypto_prices.csv"
    if not os.path.exists(prices_path):
        raise FileNotFoundError("Run data_pipeline.py first.")

    df_prices = pd.read_csv(prices_path, index_col=0, parse_dates=True).sort_index()
    all_coins = list(df_prices.columns)

    main_universe = all_coins
    smaller_universe = all_coins[:20] if len(all_coins) >= 20 else all_coins

    configs = [
        {"label": "main", "coins": main_universe},
        {"label": "smaller", "coins": smaller_universe},
    ]

    rows = []
    for cfg in configs:
        sub_prices = df_prices[cfg["coins"]].copy()
        df_returns = np.log(sub_prices).diff().dropna()

        indicator_rows = []
        window = 90
        for i in range(window - 1, len(df_returns)):
            sample = df_returns.iloc[i - window + 1:i + 1]
            indicators = compute_indicators_from_returns(sample)
            if indicators is None:
                continue
            indicator_rows.append(indicators)

        ind_df = pd.DataFrame(indicator_rows)

        rows.append({
            "Universe": cfg["label"],
            "Num_Coins": len(cfg["coins"]),
            "Avg_Correlation_Mean": ind_df["Avg_Correlation"].mean(),
            "Upper_Tail_90th_Mean": ind_df["Upper_Tail_90th"].mean(),
            "Fraction_Above_06_Mean": ind_df["Fraction_Above_06"].mean(),
            "Effective_Factors_Mean": ind_df["Effective_Factors"].mean(),
            "MST_Length_Mean": ind_df["MST_Length"].mean(),
        })

    summary = pd.DataFrame(rows)
    out_path = "data/processed/universe_sensitivity_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved universe sensitivity summary to {out_path}")

    return summary


def run_portfolio_risk_comparison():
    prices_path = "data/crypto_prices.csv"
    ind_path = "data/processed/contagion_indicators.csv"

    if not os.path.exists(prices_path) or not os.path.exists(ind_path):
        raise FileNotFoundError("Run data_pipeline.py first.")

    df_prices = pd.read_csv(prices_path, index_col=0, parse_dates=True).sort_index()
    df_returns = np.log(df_prices).diff().dropna()
    indicators = pd.read_csv(ind_path, parse_dates=["Date"]).sort_values("Date")

    market = df_returns.mean(axis=1)
    market_df = pd.DataFrame({"Market_Return": market})
    market_df["Date"] = market_df.index
    market_df = market_df.reset_index(drop=True)

    merged = pd.merge(indicators, market_df, on="Date", how="inner")
    if merged.empty:
        raise ValueError("Could not merge indicators with market returns for portfolio comparison.")

    high_threshold = merged["Avg_Correlation"].quantile(0.80)
    low_threshold = merged["Avg_Correlation"].quantile(0.20)

    high_regime = merged.loc[merged["Avg_Correlation"] >= high_threshold, "Market_Return"]
    low_regime = merged.loc[merged["Avg_Correlation"] <= low_threshold, "Market_Return"]

    rows = []
    for label, series in [("High_Contagion", high_regime), ("Low_Contagion", low_regime)]:
        if len(series) < 2:
            continue
        vol = series.std()
        mean_ret = series.mean()
        drawdown = (series.cumsum() - series.cumsum().cummax()).min()
        rows.append({
            "Regime": label,
            "Obs": len(series),
            "Mean_Return": mean_ret,
            "Volatility": vol,
            "Max_Drawdown_Proxy": drawdown,
        })

    result = pd.DataFrame(rows)
    out_path = "data/processed/portfolio_risk_comparison.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved portfolio risk comparison to {out_path}")

    return result


def run_robustness_checks():
    ensure_dirs()
    run_rolling_window_robustness()
    run_spearman_robustness()
    run_universe_sensitivity()
    run_portfolio_risk_comparison()


if __name__ == "__main__":
    run_robustness_checks()