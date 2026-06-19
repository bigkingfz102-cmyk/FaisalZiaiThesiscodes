from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
from thesis_pipeline import CONSTRUCT_LABELS, CONSTRUCT_ORDER, EXPLANATION_TYPES, prepare_data, save_figure, setup_plot_style
FIGURE_FILENAME = 'fig3_heatmap_construct_by_explanation_type.png'

def plot_construct_heatmap(means_by_construct):
    matrix = means_by_construct.pivot(index='construct', columns='explanation_type', values='mean_rating').reindex(index=CONSTRUCT_ORDER, columns=EXPLANATION_TYPES)
    matrix.index = [CONSTRUCT_LABELS[c] for c in matrix.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlGnBu', vmin=1, vmax=7, linewidths=0.5, cbar_kws={'label': 'Mean rating'}, ax=ax)
    ax.set_title('Mean Ratings per Construct and Explanation Type')
    ax.set_xlabel('Explanation type')
    ax.set_ylabel('Construct')
    fig.tight_layout()
    save_figure(fig, FIGURE_FILENAME)
    return fig

def main():
    setup_plot_style()
    _, _, _, tables, _ = prepare_data()
    plot_construct_heatmap(tables['means_by_type_and_construct.csv'])
    plt.show()
if __name__ == '__main__':
    main()
