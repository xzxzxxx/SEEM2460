from data_pipeline import run_data_pipeline
from robustness import run_robustness_checks
from visualizations import run_visualizations


def main():
    print("=" * 60)
    print("RUNNING FULL PROJECT PIPELINE")
    print("=" * 60)

    run_data_pipeline()
    run_robustness_checks()
    run_visualizations()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()