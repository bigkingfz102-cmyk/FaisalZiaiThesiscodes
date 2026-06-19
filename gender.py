from __future__ import annotations
import warnings
import matplotlib.pyplot as plt
from thesis_pipeline import OUTPUT_DIR, compute_demographics, draw_demographic_bars, ensure_output_dir, prepare_data, save_figure, setup_plot_style
GENDER_FIGURE_FILENAME = 'fig_demographics_gender_distribution.png'
GENDER_TABLE_FILENAME = 'demographic_gender_counts.csv'

def plot_gender_distribution(gender_counts):
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_demographic_bars(ax, gender_counts, 'gender', 'Gender Distribution of Participants', 'Gender', 'Greens_d')
    fig.tight_layout()
    save_figure(fig, GENDER_FIGURE_FILENAME)
    return fig

def main():
    setup_plot_style()
    *_, cleaned_df = prepare_data()
    demographics = compute_demographics(cleaned_df)
    if not demographics['has_gender']:
        warnings.warn('No gender column / gender answers found; gender graph was skipped.')
        return
    ensure_output_dir()
    gender_counts = demographics['gender_counts']
    gender_counts.to_csv(OUTPUT_DIR / GENDER_TABLE_FILENAME, index=False)
    plot_gender_distribution(gender_counts)
    print('\nGender distribution:')
    for _, row in gender_counts.iterrows():
        print(f"  {row['gender']}: {int(row['count'])} participants ({row['percentage']:.1f}%)")
    plt.show()
if __name__ == '__main__':
    main()
