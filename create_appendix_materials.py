from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from thesis_pipeline import (
    EXPLANATION_TYPES,
    EXPLANATION_TYPE_NAMES,
    OUTPUT_DIR,
    SCRIPT_DIR,
    TOPIC_NAMES,
    prepare_data,
    setup_plot_style,
)
from figure_6_topic_heatmap import plot_topic_heatmap
from figure_7_rating_distribution import plot_rating_distribution
from figure_8_scenario_heatmap import plot_scenario_heatmap
from figA4_question_selection_frequency import create_figure_a4
from figA5_explanation_exposure_count import create_figure_a5

APPENDIX_DIR = SCRIPT_DIR / "appendix"

SUBFOLDERS = {
    "A": "Appendix_A_symptom_questions",
    "B": "Appendix_B_example_explanation_set",
    "C": "Appendix_C_survey_items",
    "D": "Appendix_D_additional_figures",
    "E": "Appendix_E_statistical_tables",
    "F": "Appendix_F_data_cleaning_overview",
    "G": "Appendix_G_source_grounding",
}

SCENARIO_LETTERS = ["H", "B", "C", "F", "N", "S"]

ADDITIONAL_FIGURES = {
    "fig6_topic_by_explanation_type_heatmap.png": "topic",
    "fig8_mean_rating_by_scenario_and_type.png": "scenario",
    "fig7_rating_distribution_by_explanation_type.png": "rating",
}

STATISTICAL_TABLES = [
    "participant_level_means.csv",
    "participant_level_means_by_construct.csv",
    "pairwise_wilcoxon_tests.csv",
    "pairwise_preference_counts.csv",
    "pairwise_preference_matrix.csv",
    "preference_counts_by_type.csv",
    "means_by_type_and_construct.csv",
    "means_by_topic_and_type.csv",
    "means_by_scenario_and_type.csv",
]

SOURCE_FILE_KEYWORDS = ["thuisarts", "nhs", "source", "grounding"]

created_files = []


def folder(letter):
    return APPENDIX_DIR / SUBFOLDERS[letter]


def make_folders():
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    for letter in SUBFOLDERS:
        folder(letter).mkdir(parents=True, exist_ok=True)


def record(path):
    created_files.append(path)
    print(f"Created: {path}")


def write_table(path, columns, rows):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")
    record(path)


def all_scenarios():
    return [f"{letter}{number}" for letter in SCENARIO_LETTERS for number in range(1, 6)]


def topic_for(scenario):
    return TOPIC_NAMES.get(scenario[0], scenario[0])


def build_symptom_questions():
    rows = [
        {"scenario": scenario, "topic": topic_for(scenario), "question_text": ""}
        for scenario in all_scenarios()
    ]
    path = folder("A") / "symptom_questions.csv"
    write_table(path, ["scenario", "topic", "question_text"], rows)


def build_example_explanation_set():
    example_scenario = "H1"
    rows = [
        {
            "scenario": example_scenario,
            "explanation_type": explanation_type,
            "explanation_type_name": EXPLANATION_TYPE_NAMES[explanation_type],
            "explanation_text": "",
        }
        for explanation_type in EXPLANATION_TYPES
    ]
    path = folder("B") / "example_explanation_set_template.csv"
    write_table(
        path,
        ["scenario", "explanation_type", "explanation_type_name", "explanation_text"],
        rows,
    )


def build_survey_items():
    rows = [
        {
            "item_type": "rating item",
            "construct": "trust",
            "question_text": "I trust this explanation.",
            "scale": "1-7 Likert scale",
        },
        {
            "item_type": "rating item",
            "construct": "satisfaction",
            "question_text": "I am satisfied with this explanation.",
            "scale": "1-7 Likert scale",
        },
        {
            "item_type": "rating item",
            "construct": "usability_understandability",
            "question_text": "This explanation was easy to use and understand.",
            "scale": "1-7 Likert scale",
        },
        {
            "item_type": "rating item",
            "construct": "perceived_effectiveness",
            "question_text": "This explanation helped me understand what I should do next.",
            "scale": "1-7 Likert scale",
        },
        {
            "item_type": "validation item",
            "construct": "did_not_have_this_one",
            "question_text": "I did not have this one(fill in 1)",
            "scale": "validation only",
        },
        {
            "item_type": "preference item",
            "construct": "preference",
            "question_text": "Which of the two explanations you got do you prefer?",
            "scale": "categorical",
        },
        {
            "item_type": "demographic item",
            "construct": "age",
            "question_text": "What is your age?",
            "scale": "categorical",
        },
        {
            "item_type": "demographic item",
            "construct": "gender",
            "question_text": "What is your gender?",
            "scale": "categorical",
        },
    ]
    path = folder("C") / "survey_items.csv"
    write_table(path, ["item_type", "construct", "question_text", "scale"], rows)


def regenerate_missing_figures(missing_keys):
    setup_plot_style()
    _, ratings_df, _, tables, _ = prepare_data()
    if "fig6_topic_by_explanation_type_heatmap.png" in missing_keys:
        plot_topic_heatmap(tables["means_by_topic_and_type.csv"])
    if "fig8_mean_rating_by_scenario_and_type.png" in missing_keys:
        plot_scenario_heatmap(tables["means_by_scenario_and_type.csv"])
    if "fig7_rating_distribution_by_explanation_type.png" in missing_keys:
        plot_rating_distribution(ratings_df)
    plt.close("all")


def copy_additional_figures():
    missing = [name for name in ADDITIONAL_FIGURES if not (OUTPUT_DIR / name).exists()]
    if missing:
        print(f"Regenerating missing figures: {', '.join(missing)}")
        regenerate_missing_figures(missing)

    for name in ADDITIONAL_FIGURES:
        source = OUTPUT_DIR / name
        if source.exists():
            destination = folder("D") / name
            shutil.copy2(source, destination)
            record(destination)
        else:
            print(f"Warning: figure not found and could not be created: {name}")


def create_new_appendix_figures():
    setup_plot_style()
    create_figure_a4()
    create_figure_a5()
    plt.close("all")

    new_files = [
        "figA4_question_selection_frequency.png",
        "question_selection_frequency.csv",
        "figA5_explanation_exposure_count.png",
        "explanation_exposure_count.csv",
    ]
    for name in new_files:
        path = folder("D") / name
        if path.exists():
            record(path)


def copy_statistical_tables():
    for name in STATISTICAL_TABLES:
        source = OUTPUT_DIR / name
        if source.exists():
            destination = folder("E") / name
            shutil.copy2(source, destination)
            record(destination)
        else:
            print(f"Warning: statistical table not found, skipped: {name}")


def build_data_cleaning_overview():
    lines = [
        "Appendix F - Data cleaning and Python analysis overview",
        "=" * 60,
        "",
        "The survey responses were exported from Qualtrics as a raw CSV file.",
        "The data were then cleaned and analysed in Python.",
        "",
        "Cleaning and processing steps:",
        "- Preview responses and incomplete responses were removed where needed.",
        "- Only ratings for explanations that participants actually received were "
        "included in the analysis.",
        "- Ratings for explanations that were not selected as received were ignored.",
        "- The validation item 'I did not have this one' was not used as a rating "
        "construct; it only checked attention and consistency.",
        "",
        "Rating constructs:",
        "- The main rating constructs were trust, satisfaction, "
        "usability/understandability, and perceived effectiveness.",
        "",
        "Statistical analysis:",
        "- Participant-level mean ratings were used for the statistical tests.",
        "- A Friedman test was used to compare the five explanation types (A-E).",
        "- Wilcoxon signed-rank tests with Holm correction were used for the "
        "post-hoc pairwise comparisons.",
        "",
        "Figures:",
        "- The main results figures are reported in the results section.",
        "- Additional topic and scenario figures were placed in the appendix.",
        "",
    ]
    path = folder("F") / "data_cleaning_overview.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    record(path)


def find_source_files():
    matches = []
    for candidate in SCRIPT_DIR.glob("*.xlsx"):
        name_lower = candidate.name.lower()
        if "dashboard" in name_lower:
            continue
        if any(keyword in name_lower for keyword in SOURCE_FILE_KEYWORDS):
            matches.append(candidate)
    return matches


def build_source_grounding():
    found = find_source_files()
    for source in found:
        destination = folder("G") / source.name
        shutil.copy2(source, destination)
        record(destination)

    rows = [
        {
            "scenario": scenario,
            "topic": topic_for(scenario),
            "source_name": "",
            "source_url": "",
            "source_note": "",
        }
        for scenario in all_scenarios()
    ]
    path = folder("G") / "source_grounding_template.csv"
    write_table(
        path,
        ["scenario", "topic", "source_name", "source_url", "source_note"],
        rows,
    )


def build_index():
    lines = [
        "Appendix index",
        "=" * 60,
        "",
        "Appendix A: Full list of symptom questions",
        "Appendix B: Example explanation set",
        "Appendix C: Survey items and Likert scale",
        "Appendix D: Additional figures",
        "  - Figure A1: Mean Rating by Symptom Topic and Explanation Type",
        "  - Figure A2: Mean Rating by Scenario and Explanation Type",
        "  - Figure A3: Distribution of Ratings by Explanation Type",
        "  - Figure A4: Question Selection Frequency",
        "  - Figure A5: Explanation Exposure Count",
        "Appendix E: Additional statistical tables",
        "Appendix F: Data cleaning and Python analysis overview",
        "Appendix G: Source grounding / medical source map",
        "",
    ]
    path = APPENDIX_DIR / "appendix_index.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    record(path)


def main():
    make_folders()
    build_symptom_questions()
    build_example_explanation_set()
    build_survey_items()
    copy_additional_figures()
    create_new_appendix_figures()
    copy_statistical_tables()
    build_data_cleaning_overview()
    build_source_grounding()
    build_index()

    print("\nAppendix build complete.")
    print(f"Total appendix files created: {len(created_files)}")
    print(f"Appendix folder: {APPENDIX_DIR}")
    print("Created appendix graph A4: Question Selection Frequency")
    print("Created appendix graph A5: Explanation Exposure Count")
    print("Updated appendix index")


if __name__ == "__main__":
    main()
