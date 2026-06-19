from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from thesis_pipeline import (
    EXPLANATION_TYPES,
    EXPLANATION_TYPE_NAMES,
    OUTPUT_DIR,
    SCRIPT_DIR,
    TYPE_COLORS,
    setup_plot_style,
)

APPENDIX_FIGURE_DIR = SCRIPT_DIR / "appendix" / "Appendix_D_additional_figures"
FIGURE_FILENAME = "figA5_explanation_exposure_count.png"
TABLE_FILENAME = "explanation_exposure_count.csv"

PREFERENCES_FILE = OUTPUT_DIR / "cleaned_preferences_long.csv"
RATINGS_FILE = OUTPUT_DIR / "cleaned_ratings_long.csv"


def count_from_preferences(preferences_df):
    shown = pd.concat(
        [preferences_df["shown_type_1"], preferences_df["shown_type_2"]],
        ignore_index=True,
    )
    return shown.dropna().astype(str).str.strip().value_counts()


def count_from_ratings(ratings_df):
    unique = ratings_df.drop_duplicates(
        subset=["participant_id", "scenario", "explanation_code"]
    )
    return unique["explanation_type"].astype(str).str.strip().value_counts()


def load_exposure_counts():
    if PREFERENCES_FILE.exists():
        preferences_df = pd.read_csv(PREFERENCES_FILE)
        if {"shown_type_1", "shown_type_2"}.issubset(preferences_df.columns):
            print(f"Using {PREFERENCES_FILE.name} for exposure counts.")
            return count_from_preferences(preferences_df)

    if RATINGS_FILE.exists():
        ratings_df = pd.read_csv(RATINGS_FILE)
        if {"participant_id", "scenario", "explanation_code", "explanation_type"}.issubset(
            ratings_df.columns
        ):
            print(f"Using {RATINGS_FILE.name} for exposure counts.")
            return count_from_ratings(ratings_df)

    return None


def build_exposure_table(counts):
    rows = [
        {
            "explanation_type": explanation_type,
            "explanation_type_name": EXPLANATION_TYPE_NAMES[explanation_type],
            "times_shown": int(counts.get(explanation_type, 0)),
        }
        for explanation_type in EXPLANATION_TYPES
    ]
    return pd.DataFrame(rows)


def plot_exposure_counts(table_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    types = table_df["explanation_type"].tolist()
    values = table_df["times_shown"].tolist()
    colors = [TYPE_COLORS[explanation_type] for explanation_type in types]

    bars = ax.bar(types, values, color=colors, edgecolor="black", linewidth=0.6)

    label_offset = max(values) * 0.01 if values and max(values) > 0 else 0.1
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Explanation type")
    ax.set_ylabel("Number of times shown")
    ax.set_title("Explanation Exposure Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def create_figure_a5():
    APPENDIX_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    counts = load_exposure_counts()
    if counts is None:
        print(
            "Warning: no cleaned preference or rating data found; figure A5 was skipped."
        )
        return None

    table_df = build_exposure_table(counts)
    table_path = APPENDIX_FIGURE_DIR / TABLE_FILENAME
    table_df.to_csv(table_path, index=False, encoding="utf-8-sig")

    fig = plot_exposure_counts(table_df)
    figure_path = APPENDIX_FIGURE_DIR / FIGURE_FILENAME
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")

    print(f"Saved figure: {figure_path}")
    print(f"Saved table: {table_path}")
    return fig


def main():
    setup_plot_style()
    fig = create_figure_a5()
    if fig is not None:
        print("Created appendix graph A5: Explanation Exposure Count")
        plt.show()


if __name__ == "__main__":
    main()
