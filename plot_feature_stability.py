from math import ceil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_feature_stability(
        df: pd.DataFrame,
        features: list[str],
        n_cols: int = 4
    ):
    """
    Generates stability plots for a list of specified features from provided DataFrame.
    :param df: The input pandas DataFrame from which data is to be plotted. 
        Must contain 'ReportDate' column.
    :param features: The list of the features for which the plots will be generated.
    :param n_cols: The number of columns in the plot grid.
    """
    df_modelling = df.copy()
    df_modelling['ReportDate'] = pd.to_datetime(df_modelling['ReportDate'])
    df_modelling["DateMonth"] = df_modelling["ReportDate"].dt.to_period("M")
    df_modelling = df_modelling[["DateMonth"] + features]

    # Create bins for each numerical feature based on ranks
    for feature in features:
        ranks = df_modelling[feature].rank(method="first")
        quantile_edges = ranks.quantile([0, 0.25, 0.5, 0.75, 1]).unique()
        bins = np.concatenate(([-np.inf], quantile_edges[1:-1], [np.inf]))
        labels = ["25", "50", "75", "100"][:len(bins)-1]
        df_modelling["BIN_" + feature] = pd.cut(
            ranks,
            bins=bins,
            labels=labels,
            include_lowest=True
        )

    num_features = len(features)
    n_rows = ceil(num_features / n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 5 * n_rows))
    fig.suptitle("Feature Stability Plots", fontsize=16)

    # Plot each feature
    for i, feature in enumerate(features):
        ax = axes.flatten()[i]
        data = df_modelling.groupby("DateMonth")["BIN_" + feature].value_counts(normalize=True)
        data = data.unstack(level="BIN_" + feature).fillna(0)
        data.plot(kind='bar', stacked=True, ax=ax)
        ax.tick_params(axis='x', rotation=90)
        ax.set_title(feature)
        ax.set_ylabel('Proportion')
        ax.legend(title='Quartile')

    # Hide unused subplots
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
