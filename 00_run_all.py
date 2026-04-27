import os
import subprocess
import sys


def run_script(script_name):
    """
    Run a Python script and stop if it fails.
    """
    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print(f"{'=' * 60}\n")

    result = subprocess.run([sys.executable, script_name])

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")

    print(f"\nFinished: {script_name}\n")


def main():
    """
    Run the full crypto contagion pipeline in order.
    """

    scripts = [
        "01_fetch_data.py",
        "02_compute_correlations.py",
        "03_define_event_based_stress.py",
        "04_define_data_driven_stress.py",
        "05_compute_indicators.py",
        "06_compare_calm_stress.py",
        "07_run_rolling_window_robustness.py",
        "08_run_spearman_robustness.py",
        "09_portfolio_risk_linkage.py",
        "10_plot_results.py",
    ]

    for script in scripts:
        if not os.path.exists(script):
            raise FileNotFoundError(f"Cannot find {script}")
        run_script(script)

    print("\nAll scripts completed successfully!")


if __name__ == "__main__":
    main()