from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from thesis_pipeline import EXPLANATION_TYPES, SHORT_TYPE_LABELS, TYPE_COLORS, prepare_data, save_figure, setup_plot_style
FIGURE_FILENAME = 'fig1_overall_mean_rating_by_explanation_type.png'

def plot_overall_means(overall_means):
    plot_df = overall_means.set_index('explanation_type').reindex(EXPLANATION_TYPES)
    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(len(EXPLANATION_TYPES))
    colors = [TYPE_COLORS[t] for t in EXPLANATION_TYPES]
    ax.bar(x_positions, plot_df['mean_rating'], yerr=[plot_df['mean_rating'] - plot_df['ci95_low'], plot_df['ci95_high'] - plot_df['mean_rating']], capsize=5, color=colors, edgecolor='black', linewidth=0.6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([SHORT_TYPE_LABELS[t] for t in EXPLANATION_TYPES], rotation=15, ha='right')
    ax.set_ylim(1, 7)
    ax.set_ylabel('Mean rating (1-7)')
    ax.set_title('Overall Mean Rating by Explanation Type')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, FIGURE_FILENAME)
    return fig

def main():
    setup_plot_style()
    _, _, _, tables, _ = prepare_data()
    plot_overall_means(tables['overall_means_by_explanation_type.csv'])
    plt.show()
if __name__ == '__main__':
    main()
