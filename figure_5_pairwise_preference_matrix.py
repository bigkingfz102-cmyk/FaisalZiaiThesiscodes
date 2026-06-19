from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from thesis_pipeline import EXPLANATION_TYPES, prepare_data, save_figure, setup_plot_style
FIGURE_FILENAME = 'fig5_pairwise_preference_matrix.png'

def plot_pairwise_preference_matrix(matrix, counts_df):
    annot_labels = pd.DataFrame('', index=matrix.index, columns=matrix.columns)
    for type_1 in EXPLANATION_TYPES:
        for type_2 in EXPLANATION_TYPES:
            if type_1 == type_2:
                continue
            value = matrix.loc[type_1, type_2]
            if pd.isna(value):
                continue
            pair_row = counts_df.loc[(counts_df['type_1'] == type_1) & (counts_df['type_2'] == type_2) | (counts_df['type_1'] == type_2) & (counts_df['type_2'] == type_1)]
            if len(pair_row) == 0:
                n_decided = 0
            else:
                pair_info = pair_row.iloc[0]
                n_decided = int(pair_info['type_1_wins'] + pair_info['type_2_wins'])
            annot_labels.loc[type_1, type_2] = f'{value:.0f}%\n(n={n_decided})'
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, annot=annot_labels.values, fmt='', cmap='RdYlGn', vmin=0, vmax=100, linewidths=0.5, cbar_kws={'label': '% row preferred over column'}, ax=ax, mask=np.eye(len(EXPLANATION_TYPES), dtype=bool))
    ax.set_title('Pairwise Preference Matrix')
    ax.set_xlabel('Comparison explanation type')
    ax.set_ylabel('Preferred explanation type')
    fig.tight_layout()
    save_figure(fig, FIGURE_FILENAME)
    return fig

def main():
    setup_plot_style()
    _, _, _, tables, _ = prepare_data()
    plot_pairwise_preference_matrix(tables['pairwise_preference_matrix.csv'], tables['pairwise_preference_counts.csv'])
    plt.show()
if __name__ == '__main__':
    main()
