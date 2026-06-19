from __future__ import annotations
import warnings
import matplotlib.pyplot as plt
from thesis_pipeline import OUTPUT_DIR, build_data_quality_summary, compute_demographics, prepare_data, print_demographic_summary, print_final_summary, run_statistical_tests, save_demographic_summary, save_demographic_tables, setup_plot_style
from figure_1_overall_mean_rating import plot_overall_means
from figure_2_mean_rating_by_construct import plot_means_by_construct
from figure_3_construct_heatmap import plot_construct_heatmap
from figure_4_preference_percentage import plot_preference_percentages
from figure_5_pairwise_preference_matrix import plot_pairwise_preference_matrix
from figure_6_topic_heatmap import plot_topic_heatmap
from figure_7_rating_distribution import plot_rating_distribution
from figure_8_scenario_heatmap import plot_scenario_heatmap
from age import plot_age_distribution
from gender import plot_gender_distribution

def save_all_tables(ratings_df, preferences_df, tables, cleaning_info):
    ratings_df.to_csv(OUTPUT_DIR / 'cleaned_ratings_long.csv', index=False)
    preferences_df.to_csv(OUTPUT_DIR / 'cleaned_preferences_long.csv', index=False)
    for filename, table in tables.items():
        table.to_csv(OUTPUT_DIR / filename, index=True if filename.endswith('matrix.csv') else False)
    quality_summary = build_data_quality_summary(cleaning_info, ratings_df, preferences_df)
    quality_summary.to_csv(OUTPUT_DIR / 'data_quality_summary.csv', index=False)

def create_all_figures(tables, ratings_df):
    setup_plot_style()
    plot_overall_means(tables['overall_means_by_explanation_type.csv'])
    plot_means_by_construct(tables['means_by_type_and_construct.csv'])
    plot_construct_heatmap(tables['means_by_type_and_construct.csv'])
    plot_preference_percentages(tables['preference_counts_by_type.csv'])
    plot_pairwise_preference_matrix(tables['pairwise_preference_matrix.csv'], tables['pairwise_preference_counts.csv'])
    plot_topic_heatmap(tables['means_by_topic_and_type.csv'])
    plot_rating_distribution(ratings_df)
    plot_scenario_heatmap(tables['means_by_scenario_and_type.csv'])
    plt.show()

def create_demographic_outputs(cleaned_df):
    demographics = compute_demographics(cleaned_df)
    save_demographic_tables(demographics)
    save_demographic_summary(demographics)
    plot_age_distribution(demographics['age_counts'])
    if demographics['has_gender']:
        plot_gender_distribution(demographics['gender_counts'])
    else:
        warnings.warn('No gender data found; gender graph was skipped.')
    print_demographic_summary(demographics)
    plt.show()

def main():
    cleaning_info, ratings_df, preferences_df, tables, cleaned_df = prepare_data()
    save_all_tables(ratings_df, preferences_df, tables, cleaning_info)
    if len(ratings_df) > 0:
        create_all_figures(tables, ratings_df)
    else:
        warnings.warn('No valid rating rows found; figures were skipped.')
    setup_plot_style()
    create_demographic_outputs(cleaned_df)
    if len(ratings_df) > 0:
        run_statistical_tests(ratings_df, tables)
    else:
        warnings.warn('No valid rating rows found; statistical tests were skipped.')
    print_final_summary(cleaning_info, ratings_df, preferences_df, tables)
if __name__ == '__main__':
    main()
