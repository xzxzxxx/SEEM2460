import os
import pandas as pd


def define_event_based_stress_windows():
    """
    Create event-based stress windows around known crypto market shocks
    and save them to data/processed/stress_windows.csv.
    """

    price_path = "data/crypto_prices.csv"
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Cannot find {price_path}. Run 01_fetch_data.py first.")

    df_prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    dates = pd.DatetimeIndex(df_prices.index).sort_values()

    # Define known events: center date, pre-days, post-days
    events = [
        {"event_name": "2021_May_Selloff",      "center_date": "2021-05-19", "pre_days": 10, "during_days": 5, "post_days": 20},
        {"event_name": "2021_China_Ban",        "center_date": "2021-09-24", "pre_days": 10, "during_days": 5, "post_days": 20},
        {"event_name": "2021_Year_End_Selloff", "center_date": "2021-12-04", "pre_days": 10, "during_days": 5, "post_days": 20},

        {"event_name": "2022_Jan_Drawdown",     "center_date": "2022-01-24", "pre_days": 10, "during_days": 5, "post_days": 20},
        {"event_name": "LUNA_Collapse",         "center_date": "2022-05-12", "pre_days": 10, "during_days": 5, "post_days": 30},
        {"event_name": "Celsius_3AC_Stress",    "center_date": "2022-06-13", "pre_days": 10, "during_days": 5, "post_days": 30},
        {"event_name": "2022_Fall_Selloff",     "center_date": "2022-09-13", "pre_days": 10, "during_days": 5, "post_days": 20},
        {"event_name": "FTX_Collapse",          "center_date": "2022-11-08", "pre_days": 10, "during_days": 5, "post_days": 30},

        {"event_name": "2023_Banking_Stress",   "center_date": "2023-03-10", "pre_days": 10, "during_days": 5, "post_days": 20},
    ]

    stress_rows = []

    for event in events:
        center = pd.Timestamp(event["center_date"])
        start = center - pd.Timedelta(days=event["pre_days"])
        end = center + pd.Timedelta(days=event["post_days"])

        window_dates = dates[(dates >= start) & (dates <= end)]

        for d in window_dates:
            stress_rows.append({
                "Date": d,
                "Window_Type": "event",
                "Stress_Label": 1,
                "Event_Name": event["event_name"],
                "Event_Center": center,
                "Window_Start": start,
                "Window_End": end
            })

    df_stress = pd.DataFrame(stress_rows)

    # If a date appears in multiple event windows, keep unique rows
    if not df_stress.empty:
        df_stress = df_stress.drop_duplicates(subset=["Date", "Event_Name"])
        df_stress = df_stress.sort_values(["Date", "Event_Name"])

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/event_stress_windows.csv"
    df_stress.to_csv(output_path, index=False)

    print(f"Saved event-based stress windows to {output_path}")
    print(f"Number of stressed date-event rows: {len(df_stress)}")

    return df_stress


if __name__ == "__main__":
    define_event_based_stress_windows()