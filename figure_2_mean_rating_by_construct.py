from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from thesis_pipeline import CONSTRUCT_LABELS, CONSTRUCT_ORDER, EXPLANATION_TYPES, prepare_data, save_figure, setup_plot_style
FIGURE_FILENAME = 'fig2_mean_rating_by_construct_and_explanation_type.png'

def plot_means_by_construct(means_by_construct):
    plot_df = means_by_construct.copy()
    plot_df['construct_label'] = plot_df['construct'].map(CONSTRUCT_LABELS)
    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = np.arange(len(EXPLANATION_TYPES))
    bar_width = 0.18
    for index, construct in enumerate(CONSTRUCT_ORDER):
        subset = plot_df.loc[plot_df['construct'] == construct].set_index('explanation_type')
        subset = subset.reindex(EXPLANATION_TYPES)
        offsets = x_positions + (index - 1.5) * bar_width
        ax.bar(offsets, subset['mean_rating'], width=bar_width, label=CONSTRUCT_LABELS[construct])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(EXPLANATION_TYPES)
    ax.set_ylim(1, 7)
    ax.set_xlabel('Explanation type')
    ax.set_ylabel('Mean rating (1-7)')
    ax.set_title('Mean Rating by Construct and Explanation Type')
    ax.legend(title='Construct', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, FIGURE_FILENAME)
    return fig

def main():
    setup_plot_style()
    _, _, _, tables, _ = prepare_data()
    plot_means_by_construct(tables['means_by_type_and_construct.csv'])
    plt.show()
if __name__ == '__main__':
    main()
