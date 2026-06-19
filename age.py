from __future__ import annotations
import matplotlib.pyplot as plt
from thesis_pipeline import OUTPUT_DIR, compute_demographics, draw_demographic_bars, ensure_output_dir, prepare_data, save_figure, setup_plot_style
AGE_FIGURE_FILENAME = 'fig_demographics_age_distribution.png'
AGE_TABLE_FILENAME = 'demographic_age_counts.csv'

def plot_age_distribution(age_counts):
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_demographic_bars(ax, age_counts, 'age_group', 'Age Distribution of Participants', 'Age group', 'Blues_d')
    fig.tight_layout()
    save_figure(fig, AGE_FIGURE_FILENAME)
    return fig

def main():
    setup_plot_style()
    *_, cleaned_df = prepare_data()
    demographics = compute_demographics(cleaned_df)
    ensure_output_dir()
    age_counts = demographics['age_counts']
    age_counts.to_csv(OUTPUT_DIR / AGE_TABLE_FILENAME, index=False)
    plot_age_distribution(age_counts)
    print('\nAge distribution:')
    for _, row in age_counts.iterrows():
        print(f"  {row['age_group']}: {int(row['count'])} participants ({row['percentage']:.1f}%)")
    plt.show()
if __name__ == '__main__':
    main()
