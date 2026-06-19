from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
from thesis_pipeline import EXPLANATION_TYPES, prepare_data, save_figure, setup_plot_style
FIGURE_FILENAME = 'fig6_topic_by_explanation_type_heatmap.png'

def plot_topic_heatmap(means_by_topic):
    matrix = means_by_topic.pivot(index='topic', columns='explanation_type', values='mean_rating').reindex(columns=EXPLANATION_TYPES)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlGnBu', vmin=1, vmax=7, linewidths=0.5, cbar_kws={'label': 'Mean rating'}, ax=ax)
    ax.set_title('Mean Rating by Symptom Topic and Explanation Type')
    ax.set_xlabel('Explanation type')
    ax.set_ylabel('Symptom topic')
    fig.tight_layout()
    save_figure(fig, FIGURE_FILENAME)
    return fig

def main():
    setup_plot_style()
    _, _, _, tables, _ = prepare_data()
    plot_topic_heatmap(tables['means_by_topic_and_type.csv'])
    plt.show()
if __name__ == '__main__':
    main()
