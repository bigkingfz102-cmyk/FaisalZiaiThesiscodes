from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
from thesis_pipeline import EXPLANATION_TYPES, prepare_data, save_figure, setup_plot_style
FIGURE_FILENAME = 'fig8_mean_rating_by_scenario_and_type.png'

def plot_scenario_heatmap(means_by_scenario):
    matrix = means_by_scenario.pivot(index='scenario', columns='explanation_type', values='mean_rating').sort_index().reindex(columns=EXPLANATION_TYPES)
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlGnBu', vmin=1, vmax=7, linewidths=0.3, cbar_kws={'label': 'Mean rating'}, ax=ax)
    ax.set_title('Mean Rating by Scenario and Explanation Type')
    ax.set_xlabel('Explanation type')
    ax.set_ylabel('Scenario')
    fig.tight_layout()
    save_figure(fig, FIGURE_FILENAME)
    return fig

def main():
    setup_plot_style()
    _, _, _, tables, _ = prepare_data()
    plot_scenario_heatmap(tables['means_by_scenario_and_type.csv'])
    plt.show()
if __name__ == '__main__':
    main()
