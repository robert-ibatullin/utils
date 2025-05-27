import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_feature_stability(
    df: pd.DataFrame,
    features: list[str],
    fig_width: float = 16,
    fig_height: float = 9,
) -> go.Figure:
    """Generates stability plots for a list of specified features from provided DataFrame.

    :param df: The input pandas DataFrame from which data is to be plotted.
        Must contain 'ReportDate' column and all features listed in the next parameter.
    :param features: The list of the features for which the plots will be generated.
    :param fig_width: The width of the figure in inches. Defaults to 16.
    :param fig_height: The height of the figure in inches. Defaults to 9.
    :return: Plotly Figure.
    """
    if "ReportDate" not in df.columns:
        msg = "The DataFrame must contain 'ReportDate' column."
        raise KeyError(msg)
    for f in features:
        if f not in df.columns:
            msg = f"The DataFrame must contain '{f}' column."
            raise KeyError(msg)
    df_modelling = df.copy()
    df_modelling["ReportDate"] = pd.to_datetime(df_modelling["ReportDate"])
    df_modelling["DateMonth"] = df_modelling["ReportDate"].dt.to_period("M")
    df_modelling = df_modelling[["DateMonth", *features]]

    # Create bins for each numerical feature based on ranks
    for feature in features:
        ranks = df_modelling[feature].rank(method="first")
        quantile_edges = ranks.quantile([0, 0.25, 0.5, 0.75, 1]).unique()
        bins = np.concatenate(([-np.inf], quantile_edges[1:-1], [np.inf]))
        labels = ["25", "50", "75", "100"][: len(bins) - 1]
        df_modelling["BIN_" + feature] = pd.cut(
            ranks,
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

    n_cols, n_rows, fig_width, fig_height = get_multiplot_dimensions(
        len(features),
        fig_width,
        fig_height,
    )
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=features)
    fig.update_layout(title_text="Feature Stability Plots", height=fig_height*100, width=fig_width*100)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, feature in enumerate(features):
        data = df_modelling.groupby("DateMonth")["BIN_" + feature].value_counts(normalize=True)
        data = data.unstack(level="BIN_" + feature).fillna(0)
        
        for j, bin_label in enumerate(data.columns):
            fig.add_trace(
                go.Bar(
                    x=data.index.astype(str),
                    y=data[bin_label],
                    name=bin_label,
                    marker_color=colors[j],
                    showlegend=i == 0  # Show legend only for the first subplot
                ),
                row=(i // n_cols) + 1,
                col=(i % n_cols) + 1
            )
    
    fig.update_xaxes(tickangle=90)
    fig.update_yaxes(title_text="Proportion")
    fig.update_layout(barmode='stack', legend_title_text="Quartile")

    return fig
