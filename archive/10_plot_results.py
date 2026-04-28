import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_contagion_indicators():
    indicators_path = "data/processed/contagion_indicators.csv"
    event_path = "data/processed/event_stress_windows.csv"
    data_stress_path = "data/processed/data_driven_stress_windows.csv"

    if not os.path.exists(indicators_path):
        raise FileNotFoundError(
            f"Cannot find {indicators_path}. Run 05_compute_indicators.py first."
        )

    print("Loading contagion indicators...")
    df = pd.read_csv(indicators_path, index_col=0, parse_dates=True).sort_index()

    required_cols = [
        "Avg_Correlation",
        "Upper_Tail_90th",
        "Fraction_Above_06",
        "Effective_Factors",
        "MST_Length",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {indicators_path}: {missing_cols}")

    event_df = None
    if os.path.exists(event_path):
        event_df = pd.read_csv(event_path, parse_dates=["Date"])
        print(f"Loaded event stress windows from {event_path}")
    else:
        print("Event stress window file not found. Plot will not include event shading.")

    data_df = None
    if os.path.exists(data_stress_path):
        data_df = pd.read_csv(data_stress_path, parse_dates=["Date"])
        print(f"Loaded data-driven stress windows from {data_stress_path}")
    else:
        print("Data-driven stress window file not found. Plot will not include data-driven shading.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(14, 18), sharex=True)
    fig.suptitle("Dynamic Contagion Indicators in Cryptocurrency Market", fontsize=16, y=0.995)

    def shade_event_windows(ax):
        if event_df is None or event_df.empty:
            return

        event_names = event_df["Event_Name"].dropna().unique()
        for i, event_name in enumerate(event_names):
            sub = event_df[event_df["Event_Name"] == event_name]
            if "Window_Start" in sub.columns and "Window_End" in sub.columns:
                start = pd.to_datetime(sub["Window_Start"].iloc[0])
                end = pd.to_datetime(sub["Window_End"].iloc[0])
                ax.axvspan(
                    start,
                    end,
                    color="gray",
                    alpha=0.15,
                    label="Event Stress" if i == 0 else None
                )

    def shade_data_stress(ax):
        if data_df is None or data_df.empty:
            return

        stress_dates = data_df.loc[data_df["Stress_Label"] == 1, "Date"]
        if stress_dates.empty:
            return

        stress_dates = pd.DatetimeIndex(stress_dates).sort_values()
        start = stress_dates[0]
        prev = stress_dates[0]
        first_block = True

        for d in stress_dates[1:]:
            if (d - prev).days > 1:
                ax.axvspan(
                    start,
                    prev,
                    color="red",
                    alpha=0.08,
                    label="Data-driven Stress" if first_block else None
                )
                first_block = False
                start = d
            prev = d

        ax.axvspan(
            start,
            prev,
            color="red",
            alpha=0.08,
            label="Data-driven Stress" if first_block else None
        )

    axes[0].plot(df.index, df["Avg_Correlation"], color="blue", linewidth=1.5, label="Avg Correlation")
    shade_event_windows(axes[0])
    shade_data_stress(axes[0])
    axes[0].set_title("1. Average Pairwise Correlation")
    axes[0].set_ylabel("Correlation")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(df.index, df["Upper_Tail_90th"], color="red", linewidth=1.5)
    shade_event_windows(axes[1])
    shade_data_stress(axes[1])
    axes[1].set_title("2. Upper-Tail Correlation (90th Percentile)")
    axes[1].set_ylabel("Correlation")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(df.index, df["Fraction_Above_06"], color="purple", linewidth=1.5)
    shade_event_windows(axes[2])
    shade_data_stress(axes[2])
    axes[2].set_title("3. Fraction of Highly Correlated Pairs (> 0.6)")
    axes[2].set_ylabel("Fraction")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    axes[3].plot(df.index, df["Effective_Factors"], color="orange", linewidth=1.5)
    shade_event_windows(axes[3])
    shade_data_stress(axes[3])
    axes[3].set_title("4. Effective Number of Factors")
    axes[3].set_ylabel("Number of Factors")
    axes[3].grid(True, linestyle="--", alpha=0.5)

    axes[4].plot(df.index, df["MST_Length"], color="green", linewidth=1.5)
    shade_event_windows(axes[4])
    shade_data_stress(axes[4])
    axes[4].set_title("5. Minimum Spanning Tree Total Length")
    axes[4].set_ylabel("Distance")
    axes[4].set_xlabel("Date")
    axes[4].grid(True, linestyle="--", alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = os.path.join(figures_dir, "contagion_dashboard.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSuccess! Dashboard saved to {output_path}")


if __name__ == "__main__":
    plot_contagion_indicators()