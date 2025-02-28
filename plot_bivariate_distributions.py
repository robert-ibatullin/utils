from math import ceil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_bivariate_distribution(
        df: pd.DataFrame,
        feature: str,
        ax: plt.axes,
    ) -> plt.axes:

    if pd.api.types.is_integer_dtype(df[feature]):
        # Use consecutive integers as bins
        min_value = df[feature].min()
        max_value = df[feature].max()
        bins = np.arange(min_value, max_value + 2) - 0.5  # Extend to include the last integer
        logscale = (min_value > 0 and max_value / min_value > 100)
    else:
        bins = 30
        logscale = False

    bin_edges = np.histogram_bin_edges(df[feature], bins=bins)
    density_0, _ = np.histogram(df[df['target'] == 0][feature], bins=bin_edges, density=True)
    density_1, _ = np.histogram(df[df['target'] == 1][feature], bins=bin_edges, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    ax.barh(bin_centers, -density_0, height=(bin_edges[1] - bin_edges[0]),
            color='blue', label='Target 0')
    ax.barh(bin_centers, density_1, height=(bin_edges[1] - bin_edges[0]),
            color='orange', label='Target 1')
    ax.plot(density_1 - density_0, bin_centers, color='black', label='Difference')

    if logscale:
        ax.set_xscale('log')

    ax.set_xlabel('Density')
    ax.set_title(feature)

    # Center the y-axis
    ax.axvline(0, color='black', linewidth=0.8)

    # Set the x-axis limits to be symmetric around the y-axis
    max_density = max(density_0.max(), density_1.max())
    ax.set_xlim(-max_density, max_density)

    ax.legend(loc='best')

    return ax


def plot_bivariate_distributions(
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        features: list[str],
        n_cols: int = 4
    ):
    num_features = len(features)
    n_rows = ceil(num_features / n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 5 * n_rows))
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

    # Hide unused subplots
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
