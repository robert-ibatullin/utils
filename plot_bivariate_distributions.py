import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from make_multiplot import get_multiplot_dimensions


def plot_bivariate_distribution(
        df: pd.DataFrame,
        feature: str,
        ax: plt.axes,
    ) -> plt.axes:

    min_value = df[feature].min()
    max_value = df[feature].max()
    logscale = (min_value >= 0 and max_value / (min_value+1) > 100)
    if not logscale:
        if pd.api.types.is_integer_dtype(df[feature]):
            # Use consecutive integers as bins
            bins = np.arange(min_value, max_value + 2) - 0.5  # Extend to include the last integer
        else:
            bins = np.linspace(min_value, max_value, 20)
    else:
        bins = np.geomspace(min_value+0.5, max_value, 20)

    bin_edges = np.histogram_bin_edges(df[feature], bins=bins)
    density_0, _ = np.histogram(df[df['target'] == 0][feature], bins=bin_edges, density=True)
    density_1, _ = np.histogram(df[df['target'] == 1][feature], bins=bin_edges, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    heights = bin_edges[1:] - bin_edges[:-1]
    ax.barh(bin_centers, -density_0,
                        height=heights,
            color='blue', label='Target 0')
    ax.barh(bin_centers, density_1,
            height=heights,
            color='orange', label='Target 1')
    ax.plot(density_1 - density_0, bin_centers, color='black', label='Difference')

    if logscale:
        ax.set_yscale('log')

    # ax.set_xlabel('Density')
    ax.set_title(feature)

    # Center the y-axis
    ax.axvline(0, color='black', linewidth=0.8)

    # Set the x-axis limits to be symmetric around the y-axis
    max_density = max(density_0.max(), density_1.max())
    ax.set_xlim(-max_density, max_density)

    return ax


def plot_bivariate_distributions(
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        features: list[str],
    ):
    num_features = len(features)
    n_cols, n_rows, fig_width, fig_height = get_multiplot_dimensions(num_features, 16, 9)
    # Create the figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    fig.suptitle("Bivariate distributions of features by target", fontsize=16)

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    df = X.copy()
    df['target'] = y

    for i, feature in enumerate(features):
        ax = axes.flatten()[i]
        ax = plot_bivariate_distribution(df, feature, ax)
    handles, labels = ax.get_legend_handles_labels()

    # Hide unused subplots
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    # Create a single legend for all subplots
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
