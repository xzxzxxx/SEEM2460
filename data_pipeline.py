import os
import pandas as pd
import numpy as np
import yfinance as yf
import networkx as nx


START_DATE = "2020-01-01"
END_DATE = "2024-01-01"

TICKERS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
    "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "LTC-USD",
    "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD", "UNI-USD",
    "XMR-USD", "ETC-USD", "BCH-USD", "XLM-USD", "ALGO-USD",
    "FIL-USD", "ICP-USD", "VET-USD", "AAVE-USD", "NEAR-USD",
    "HBAR-USD", "EGLD-USD", "SAND-USD", "MANA-USD", "THETA-USD"
]

EVENT_WINDOWS = [
    {"name": "COVID_2020", "start": "2020-03-01", "end": "2020-05-31"},
    {"name": "FTX_2022", "start": "2022-11-01", "end": "2022-12-15"},
    {"name": "ETF_2024", "start": "2024-01-01", "end": "2024-01-31"},
]


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("figures", exist_ok=True)


def fetch_crypto_data():
    print(f"Downloading data for {len(TICKERS)} cryptocurrencies...")
    print(f"Date range: {START_DATE} to {END_DATE}")

    raw = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
        threads=True
    )

    if raw.empty:
        raise ValueError("No data returned by yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        top_level = raw.columns.get_level_values(0)
        if "Adj Close" in top_level:
            df = raw["Adj Close"].copy()
        elif "Close" in top_level:
            df = raw["Close"].copy()
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' found in downloaded data.")
    else:
        if "Adj Close" in raw.columns:
            df = raw[["Adj Close"]].copy()
        elif "Close" in raw.columns:
            df = raw[["Close"]].copy()
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' found in downloaded data.")

    df.columns = [str(col).replace("-USD", "") for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_index()
    df = df.ffill()

    min_non_missing = int(len(df) * 0.8)
    before_cols = len(df.columns)
    df = df.dropna(axis=1, thresh=min_non_missing)
    after_cols = len(df.columns)
    dropped_cols = before_cols - after_cols

    df = df.dropna()

    output_path = "data/crypto_prices.csv"
    df.to_csv(output_path)

    print(f"Saved prices to {output_path}")
    print(f"Final dataset shape: {df.shape}")
    print(f"Retained coins ({after_cols}): {list(df.columns)}")
    print(f"Dropped coins due to insufficient coverage: {dropped_cols}")
    print(f"Date coverage: {df.index.min().date()} to {df.index.max().date()}")

    return df


def compute_rolling_correlations(df_prices, window_size=90):
    df_returns = np.log(df_prices).diff().dropna()
    rolling_corr = df_returns.rolling(window=window_size).corr().dropna(how="all")

    out_path = "data/processed/rolling_correlations.csv"
    rolling_corr.to_csv(out_path)
    print(f"Saved rolling correlations to {out_path}")
    print(f"Rolling window size: {window_size} days")

    return df_returns, rolling_corr


def _indicator_row_from_corr_matrix(corr_matrix):
    if corr_matrix.ndim != 2 or corr_matrix.shape[0] != corr_matrix.shape[1]:
        return None

    upper_tri_idx = np.triu_indices_from(corr_matrix, k=1)
    pairwise_corrs = corr_matrix[upper_tri_idx]
    pairwise_corrs = pairwise_corrs[~np.isnan(pairwise_corrs)]

    if len(pairwise_corrs) == 0:
        return None

    clean_matrix = np.nan_to_num(corr_matrix, nan=0.0)

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
        "Avg_Correlation": float(np.mean(pairwise_corrs)),
        "Upper_Tail_90th": float(np.percentile(pairwise_corrs, 90)),
        "Fraction_Above_06": float(np.mean(pairwise_corrs > 0.6)),
        "Effective_Factors": float(effective_factors),
        "MST_Length": float(mst_length),
    }


def compute_contagion_indicators(df_prices, window_size=90):
    df_returns = np.log(df_prices).diff().dropna()
    rolling_corr = df_returns.rolling(window=window_size).corr().dropna(how="all")
    dates = rolling_corr.index.get_level_values(0).unique()

    rows = []
    for date in dates:
        corr_matrix = rolling_corr.loc[date].values
        indicators = _indicator_row_from_corr_matrix(corr_matrix)
        if indicators is None:
            continue
        rows.append({"Date": date, **indicators})

    df_indicators = pd.DataFrame(rows).set_index("Date").sort_index()

    out_path = "data/processed/contagion_indicators.csv"
    df_indicators.to_csv(out_path)

    print(f"Saved contagion indicators to {out_path}")
    print(f"Indicator rows: {len(df_indicators)}")
    print("Indicator columns:", list(df_indicators.columns))
    print(df_indicators.describe().T[["mean", "std", "min", "max"]])

    return df_indicators


def define_event_based_stress(df_prices):
    df_returns = np.log(df_prices).diff().dropna()

    rows = []
    for event in EVENT_WINDOWS:
        mask = (
            (df_returns.index >= pd.to_datetime(event["start"])) &
            (df_returns.index <= pd.to_datetime(event["end"]))
        )
        event_returns = df_returns.loc[mask]

        for dt in event_returns.index:
            rows.append({
                "Date": dt,
                "Event": event["name"],
                "Stress_Label": 1
            })

    stress_df = pd.DataFrame(rows).drop_duplicates(subset=["Date"]).sort_values("Date")
    out_path = "data/processed/event_stress_windows.csv"
    stress_df.to_csv(out_path, index=False)

    print(f"Saved event stress windows to {out_path}")
    print(f"Event-stress dates: {len(stress_df)}")
    print(stress_df["Event"].value_counts())

    return stress_df


def define_data_driven_stress(df_indicators):
    threshold = df_indicators["Avg_Correlation"].quantile(0.90)

    stress_df = df_indicators.reset_index()[["Date", "Avg_Correlation"]].copy()
    stress_df["Stress_Label"] = (stress_df["Avg_Correlation"] >= threshold).astype(int)

    out_path = "data/processed/data_driven_stress_windows.csv"
    stress_df.to_csv(out_path, index=False)

    print(f"Saved data-driven stress windows to {out_path}")
    print(f"Data-driven stress threshold (90th percentile Avg_Correlation): {threshold:.4f}")
    print(f"Data-driven stress dates: {int(stress_df['Stress_Label'].sum())}")

    return stress_df


def stress_window_summary(df_indicators, event_stress_df, data_driven_stress_df):
    ind = df_indicators.reset_index().copy()
    ind["Date"] = pd.to_datetime(ind["Date"])
    ind["Date_norm"] = ind["Date"].dt.normalize()

    event_dates = set(pd.to_datetime(event_stress_df["Date"]).dt.normalize())
    data_stress_dates = set(
        pd.to_datetime(
            data_driven_stress_df.loc[data_driven_stress_df["Stress_Label"] == 1, "Date"]
        ).dt.normalize()
    )

    ind["Regime"] = "Calm"
    ind.loc[ind["Date_norm"].isin(event_dates), "Regime"] = "Event_Stress"
    ind.loc[ind["Date_norm"].isin(data_stress_dates), "Regime"] = "Data_Stress"
    ind.loc[ind["Date_norm"].isin(event_dates) | ind["Date_norm"].isin(data_stress_dates), "Regime"] = "Stress"

    summary = ind.groupby("Regime")[[
        "Avg_Correlation",
        "Upper_Tail_90th",
        "Fraction_Above_06",
        "Effective_Factors",
        "MST_Length"
    ]].agg(["mean", "median", "std", "count"])

    out_path = "data/processed/stress_window_summary.csv"
    summary.to_csv(out_path)

    print(f"Saved stress window summary to {out_path}")
    print("\n=== Stress Window Summary ===")
    print(summary)

    return summary


def compare_calm_vs_stress(df_indicators, event_stress_df, data_driven_stress_df):
    ind = df_indicators.reset_index().copy()
    ind["Date"] = pd.to_datetime(ind["Date"])

    event_dates = set(pd.to_datetime(event_stress_df["Date"]).dt.normalize())
    data_stress_dates = set(
        pd.to_datetime(
            data_driven_stress_df.loc[data_driven_stress_df["Stress_Label"] == 1, "Date"]
        ).dt.normalize()
    )

    ind["Date_norm"] = ind["Date"].dt.normalize()
    ind["Is_Stress"] = ind["Date_norm"].isin(event_dates) | ind["Date_norm"].isin(data_stress_dates)

    calm = ind.loc[~ind["Is_Stress"]]
    stress = ind.loc[ind["Is_Stress"]]

    summary = pd.DataFrame({
        "Metric": [
            "Avg_Correlation",
            "Upper_Tail_90th",
            "Fraction_Above_06",
            "Effective_Factors",
            "MST_Length",
        ],
        "Calm_Mean": [
            calm["Avg_Correlation"].mean(),
            calm["Upper_Tail_90th"].mean(),
            calm["Fraction_Above_06"].mean(),
            calm["Effective_Factors"].mean(),
            calm["MST_Length"].mean(),
        ],
        "Stress_Mean": [
            stress["Avg_Correlation"].mean(),
            stress["Upper_Tail_90th"].mean(),
            stress["Fraction_Above_06"].mean(),
            stress["Effective_Factors"].mean(),
            stress["MST_Length"].mean(),
        ],
    })

    out_path = "data/processed/calm_stress_comparison.csv"
    summary.to_csv(out_path, index=False)

    print(f"Saved calm vs stress comparison to {out_path}")
    print("\n=== Calm vs Stress Comparison ===")
    print(summary)

    return summary


def run_data_pipeline():
    ensure_dirs()
    df_prices = fetch_crypto_data()
    _, _ = compute_rolling_correlations(df_prices, window_size=90)
    df_indicators = compute_contagion_indicators(df_prices, window_size=90)
    event_stress_df = define_event_based_stress(df_prices)
    data_driven_stress_df = define_data_driven_stress(df_indicators)
    stress_window_summary(df_indicators, event_stress_df, data_driven_stress_df)
    compare_calm_vs_stress(df_indicators, event_stress_df, data_driven_stress_df)


if __name__ == "__main__":
    run_data_pipeline()