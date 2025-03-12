import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from math import ceil
from typing import Callable


def get_multiplot_dimensions(num_features: int, width: int, height: int) -> tuple[int, int, float, float]:
    if num_features <= width / 3:
        # All subplots fit in one line
        n_cols = num_features
        n_rows = 1
        subplot_width = width / num_features
        subplot_height = subplot_width / (width / height)
    else:
        # Subplots need more than one line
        n_cols = int(width / 3)
        subplot_width = 3  # Minimum subplot width
        subplot_height = subplot_width / (width / height)
        n_rows = ceil(num_features / n_cols)
    fig_width = n_cols * subplot_width
    fig_height = (1+n_rows) * subplot_height
    return n_cols, n_rows, fig_width, fig_height
