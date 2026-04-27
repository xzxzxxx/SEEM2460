import os
import pandas as pd
from scipy.stats import mannwhitneyu


def compare_calm_vs_stress():
    """
    Compare contagion indicators between calm and stress periods.
    """

    indicators_path = "data/processed/contagion_indicators.csv"
    event_path = "data/processed/event_stress_windows.csv"
    data_driven_path = "data/processed/data_driven_stress_windows.csv"

    if not os.path.exists(indicators_path):
        raise FileNotFoundError(f"Cannot find {indicators_path}. Run 04_compute_indicators.py first.")

    df_ind = pd.read_csv(indicators_path, index_col=0, parse_dates=True)
    df_ind = df_ind.sort_index()
    df_ind = df_ind.reset_index().rename(columns={"index": "Date"})

    # Load event-based stress labels
    if os.path.exists(event_path):
        df_event = pd.read_csv(event_path, parse_dates=["Date"])
        df_event["Event_Stress"] = 1
    else:
        df_event = pd.DataFrame(columns=["Date", "Event_Stress"])

    # Load data-driven stress labels
    if os.path.exists(data_driven_path):
        df_data = pd.read_csv(data_driven_path, parse_dates=["Date"])
        df_data = df_data[["Date", "Stress_Label"]].rename(columns={"Stress_Label": "Data_Stress"})
    else:
        df_data = pd.DataFrame(columns=["Date", "Data_Stress"])

    # Collapse event windows to one label per date
    if not df_event.empty:
        df_event_date = df_event.groupby("Date", as_index=False)["Event_Stress"].max()
    else:
        df_event_date = pd.DataFrame({"Date": [], "Event_Stress": []})

    # Merge everything
    df = df_ind.merge(df_event_date, on="Date", how="left")
    df = df.merge(df_data, on="Date", how="left")

    df["Event_Stress"] = df["Event_Stress"].fillna(0).astype(int)
    df["Data_Stress"] = df["Data_Stress"].fillna(0).astype(int)

    # Define calm/stress samples
    stress_cols = ["Event_Stress", "Data_Stress"]

    indicators = [
        "Avg_Correlation",
        "Upper_Tail_90th",
        "Fraction_Above_06",
        "Effective_Factors",
        "MST_Length"
    ]

    rows = []

    for ind in indicators:
        calm_vals = df.loc[(df["Event_Stress"] == 0) & (df["Data_Stress"] == 0), ind].dropna()
        event_vals = df.loc[df["Event_Stress"] == 1, ind].dropna()
        data_vals = df.loc[df["Data_Stress"] == 1, ind].dropna()

        def summarize(name, series):
            return {
                "Indicator": ind,
                "Group": name,
                "Count": len(series),
                "Mean": series.mean(),
                "Median": series.median(),
                "Std": series.std()
            }

        rows.append(summarize("Calm", calm_vals))
        rows.append(summarize("Event_Stress", event_vals))
        rows.append(summarize("Data_Stress", data_vals))

        # Optional pairwise tests
        if len(calm_vals) > 0 and len(event_vals) > 0:
            u1 = mannwhitneyu(calm_vals, event_vals, alternative="two-sided")
            rows.append({
                "Indicator": ind,
                "Group": "Calm_vs_Event_PValue",
                "Count": None,
                "Mean": None,
                "Median": None,
                "Std": None,
                "Statistic": u1.statistic,
                "PValue": u1.pvalue
            })

        if len(calm_vals) > 0 and len(data_vals) > 0:
            u2 = mannwhitneyu(calm_vals, data_vals, alternative="two-sided")
            rows.append({
                "Indicator": ind,
                "Group": "Calm_vs_Data_PValue",
                "Count": None,
                "Mean": None,
                "Median": None,
                "Std": None,
                "Statistic": u2.statistic,
                "PValue": u2.pvalue
            })

    df_summary = pd.DataFrame(rows)

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/stress_comparison_summary.csv"
    df_summary.to_csv(output_path, index=False)

    print(f"Saved calm vs stress comparison summary to {output_path}")
    print(df_summary.head(15))

    return df_summary


if __name__ == "__main__":
    compare_calm_vs_stress()