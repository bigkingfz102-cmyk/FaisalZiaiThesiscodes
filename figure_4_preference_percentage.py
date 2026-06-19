from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from thesis_pipeline import EXPLANATION_TYPES, TYPE_COLORS, prepare_data, save_figure, setup_plot_style
FIGURE_FILENAME = 'fig4_preference_percentage_by_explanation_type.png'

def plot_preference_percentages(preference_counts):
    plot_df = preference_counts.set_index('explanation_type').reindex(EXPLANATION_TYPES)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [TYPE_COLORS[t] for t in EXPLANATION_TYPES]
    bars = ax.bar(EXPLANATION_TYPES, plot_df['preference_percentage'], color=colors, edgecolor='black', linewidth=0.6)
    for bar, value in zip(bars, plot_df['preference_percentage']):
        if pd.notna(value):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Explanation type')
    ax.set_ylabel('Preference percentage')
    ax.set_title('Preference Percentage by Explanation Type')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, FIGURE_FILENAME)
    return fig

def main():
    setup_plot_style()
    _, _, _, tables, _ = prepare_data()
    plot_preference_percentages(tables['preference_counts_by_type.csv'])
    plt.show()
if __name__ == '__main__':
    main()
