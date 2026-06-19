from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
from thesis_pipeline import EXPLANATION_TYPES, TYPE_COLORS, prepare_data, save_figure, setup_plot_style
FIGURE_FILENAME = 'fig7_rating_distribution_by_explanation_type.png'

def plot_rating_distribution(ratings_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=ratings_df, x='explanation_type', y='rating', order=EXPLANATION_TYPES, hue='explanation_type', hue_order=EXPLANATION_TYPES, palette=TYPE_COLORS, legend=False, inner='box', cut=0, ax=ax)
    ax.set_ylim(1, 7)
    ax.set_xlabel('Explanation type')
    ax.set_ylabel('Rating (1-7)')
    ax.set_title('Distribution of Ratings by Explanation Type')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, FIGURE_FILENAME)
    return fig

def main():
    setup_plot_style()
    _, ratings_df, _, _, _ = prepare_data()
    plot_rating_distribution(ratings_df)
    plt.show()
if __name__ == '__main__':
    main()
