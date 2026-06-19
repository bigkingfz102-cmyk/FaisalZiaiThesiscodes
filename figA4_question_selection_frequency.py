from __future__ import annotations

import re

import matplotlib.pyplot as plt
import pandas as pd

from thesis_pipeline import (
    CSV_FILENAME,
    SCRIPT_DIR,
    TOPIC_NAMES,
    clean_participants,
    load_qualtrics_csv,
    setup_plot_style,
)

APPENDIX_FIGURE_DIR = SCRIPT_DIR / "appendix" / "Appendix_D_additional_figures"
FIGURE_FILENAME = "figA4_question_selection_frequency.png"
TABLE_FILENAME = "question_selection_frequency.csv"

SCENARIO_LETTERS = ["H", "B", "C", "F", "N", "S"]
SELECTION_ITEM_PATTERN = re.compile(r"(?:^|,)\s*([HBCFNS][1-5])\s*-\s*")


def all_scenarios():
    return [f"{letter}{number}" for letter in SCENARIO_LETTERS for number in range(1, 6)]


def find_selection_column(columns, column_map):
    for col in columns:
        if str(col).strip().lower() == "q426":
            return col
    for col in columns:
        text = column_map.get(col, "").lower()
        if "select exactly 20" in text or "predefined questions" in text:
            return col
    return None


def parse_selected_items(value):
    if pd.isna(value):
        return []
    text = str(value)
    matches = list(SELECTION_ITEM_PATTERN.finditer(text))
    items = []
    for index, match in enumerate(matches):
        scenario = match.group(1).upper()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        question_text = text[start:end].strip().strip(",").strip()
        items.append((scenario, question_text))
    return items


def build_selection_counts(cleaned_df, selection_col):
    counts = {scenario: 0 for scenario in all_scenarios()}
    question_texts = {}

    for value in cleaned_df[selection_col]:
        scenarios_seen = set()
        for scenario, question_text in parse_selected_items(value):
            if scenario not in counts:
                continue
            if question_text and scenario not in question_texts:
                question_texts[scenario] = question_text
            if scenario not in scenarios_seen:
                counts[scenario] += 1
                scenarios_seen.add(scenario)

    rows = [
        {
            "scenario": scenario,
            "topic": TOPIC_NAMES.get(scenario[0], scenario[0]),
            "question_text": question_texts.get(scenario, ""),
            "selection_count": counts[scenario],
        }
        for scenario in all_scenarios()
    ]
    return pd.DataFrame(rows)


def plot_selection_frequency(counts_df):
    fig, ax = plt.subplots(figsize=(15, 6))
    scenarios = counts_df["scenario"].tolist()
    values = counts_df["selection_count"].tolist()

    bars = ax.bar(scenarios, values, color="#4C72B0", edgecolor="black", linewidth=0.6)

    label_offset = max(values) * 0.01 if values and max(values) > 0 else 0.1
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Number of participants who selected the question")
    ax.set_title("Question Selection Frequency")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return fig


def create_figure_a4():
    APPENDIX_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw_df, column_map = load_qualtrics_csv(SCRIPT_DIR / CSV_FILENAME)
    cleaned_df, _ = clean_participants(raw_df, column_map)

    selection_col = find_selection_column(cleaned_df.columns, column_map)
    if selection_col is None:
        print("Warning: question selection column not found; figure A4 was skipped.")
        return None

    counts_df = build_selection_counts(cleaned_df, selection_col)
    table_path = APPENDIX_FIGURE_DIR / TABLE_FILENAME
    counts_df.to_csv(table_path, index=False, encoding="utf-8-sig")

    fig = plot_selection_frequency(counts_df)
    figure_path = APPENDIX_FIGURE_DIR / FIGURE_FILENAME
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")

    print(f"Saved figure: {figure_path}")
    print(f"Saved table: {table_path}")
    return fig


def main():
    setup_plot_style()
    fig = create_figure_a4()
    if fig is not None:
        print("Created appendix graph A4: Question Selection Frequency")
        plt.show()


if __name__ == "__main__":
    main()
