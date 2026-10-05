"""Visualization utilities.

Plotly helpers (funnel, concat, parallel categories) and matplotlib subplot
helpers live here. Statistical forests are ``statract.plot_forest``;
:mod:`statract.figure.hr_forest` is a compatibility shim.
"""

from __future__ import annotations

import io
from typing import Any

from .hr_forest import (
    FigureWidthPreset,
    ForestRow,
    ForestTextCol,
    rows_from_prepared,
    rows_from_prepared_grouped,
    rows_from_prepared_named_groups,
    save_forest_panel_rows,
    save_prepared_hr_forest,
)

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from PIL import Image


def parallel_categories_keeporder(
    df: pl.DataFrame,
    dimensions: list[str],
    orders: list[list[Any]],
    **kwargs: Any,
) -> go.Figure:
    """Create parallel categories plot with custom order for each dimension.

    Args:
        df: Polars DataFrame with categorical data
        dimensions: List of column names to use as dimensions
        orders: List of lists specifying the desired order for each dimension
        **kwargs: Additional arguments passed to px.parallel_categories

    Returns:
        Plotly figure with ordered categories

    Examples:
        >>> import polars as pl
        >>> from statract.figure import parallel_categories_keeporder
        >>>
        >>> df = pl.DataFrame({
        ...     "category": ["A", "B", "C", "A", "B"],
        ...     "value": [1, 2, 3, 4, 5],
        ...     "group": ["X", "Y", "X", "Y", "X"]
        ... })
        >>> fig = parallel_categories_keeporder(
        ...     df,
        ...     dimensions=["category", "group", "value"],
        ...     orders=[["A", "B", "C"], ["X", "Y"], [1, 2, 3, 4, 5]]
        ... )
        >>> fig.show()
    """
    assert len(dimensions) == len(orders)
    options = []
    for dim, order in zip(dimensions, orders, strict=False):
        e = list(df[dim].unique())
        o = list(order)
        categoryarray = [i for i in o if i in e] + [i for i in e if i not in o]
        options.append(
            {
                "categoryorder": "array",
                "categoryarray": categoryarray,
            },
        )
    return px.parallel_categories(df, dimensions=dimensions, **kwargs).update_traces(dimensions=options)


def concat_plotly_figures(fig0: go.Figure, fig1: go.Figure) -> Image.Image:
    """Concatenate two plotly figures horizontally into a single PIL Image.

    Args:
        fig0: First plotly figure
        fig1: Second plotly figure

    Returns:
        PIL Image with both figures side by side

    Examples:
        >>> import polars as pl
        >>> import plotly.express as px
        >>> from statract.figure import concat_plotly_figures
        >>>
        >>> df1 = pl.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})
        >>> df2 = pl.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4]})
        >>> fig1 = px.scatter(df1, x="x", y="y")
        >>> fig2 = px.scatter(df2, x="x", y="y")
        >>> combined = concat_plotly_figures(fig1, fig2)
        >>> combined.save("combined.png")
    """
    png0 = io.BytesIO()
    png1 = io.BytesIO()
    pio.write_image(fig0, png0)
    pio.write_image(fig1, png1)

    i0 = Image.open(png0)
    i1 = Image.open(png1)
    i = Image.new("RGB", (i0.width + i1.width, max(i0.height, i1.height)))
    i.paste(i0, (0, 0))
    i.paste(i1, (i0.width, 0))
    return i


def concat_images(images: list[Image.Image], ncols: int = 10) -> Image.Image:
    """Concatenate multiple images into a grid.

    Args:
        images: List of PIL Images to concatenate
        ncols: Number of columns in the grid (default: 10)

    Returns:
        PIL Image with all images arranged in a grid

    Raises:
        ValueError: If images list is empty

    Examples:
        >>> from PIL import Image
        >>> from statract.figure import concat_images
        >>>
        >>> # Create sample images
        >>> img1 = Image.new("RGB", (100, 100), color="red")
        >>> img2 = Image.new("RGB", (100, 100), color="blue")
        >>> img3 = Image.new("RGB", (100, 100), color="green")
        >>>
        >>> # Concatenate into 2 columns
        >>> grid = concat_images([img1, img2, img3], ncols=2)
        >>> grid.save("grid.png")
    """
    # https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python/46623632#46623632
    n_images = len(images)
    if n_images == 0:
        raise ValueError("Cannot concatenate empty list of images")
    n_horiz = min(n_images, ncols)
    h_sizes, v_sizes = [0] * n_horiz, [0] * ((n_images + n_horiz - 1) // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new("RGB", (h_sizes[-1], v_sizes[-1]), color="white")
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid


def _get_lim(p: float, q: float) -> tuple[float, float]:
    """Calculate axis limits with small margin.

    Args:
        p: Minimum value
        q: Maximum value

    Returns:
        Tuple of (min_limit, max_limit) with 0.5% margin

    Examples:
        >>> from statract.figure import get_lim
        >>>
        >>> min_val, max_val = get_lim(0.0, 100.0)
        >>> min_val < 0.0
        True
        >>> max_val > 100.0
        True
    """
    r = q - p
    m = r * 0.005
    return (p - m, q + m)


def funnel_plot(
    df: pl.DataFrame,
    n: str,
    y: str,
    raw_y: pl.Series,
    **kwargs: Any,
) -> go.Figure:
    """Create a funnel plot with confidence limits.

    A funnel plot is used to visualize the relationship between sample size
    and outcome value, with confidence limits showing expected variation.

    Args:
        df: Polars DataFrame with data points
        n: Column name for sample size (x-axis)
        y: Column name for outcome value (y-axis)
        raw_y: Polars Series with raw outcome values for calculating mean/std
        **kwargs: Additional arguments passed to px.scatter

    Returns:
        Plotly figure with funnel plot including 99.9% confidence limits

    Examples:
        >>> import polars as pl
        >>> from statract.figure import funnel_plot
        >>>
        >>> df = pl.DataFrame({
        ...     "n": [10, 20, 30, 40, 50],
        ...     "mean": [0.5, 0.52, 0.48, 0.51, 0.49]
        ... })
        >>> raw_y = pl.Series([0.4, 0.5, 0.6, 0.45, 0.55, 0.5, 0.52, 0.48])
        >>> fig = funnel_plot(df, n="n", y="mean", raw_y=raw_y)
        >>> fig.show()
    """
    # https://stats.stackexchange.com/questions/5195/how-to-draw-funnel-plot-using-ggplot2-in-r
    fig = px.scatter(df, x=n, y=y, **kwargs)

    nm = float(df[n].max())
    lx = np.geomspace(0.1, nm, 25)

    line_option = dict(color="black", width=1, dash="dot")
    tw = dict(line=line_option, showlegend=False)

    # 1.96: 95%
    # 2.58: 99%
    # 3.29: 99.9%
    mean = float(raw_y.mean())
    std_val = raw_y.std(ddof=0)  # Use population std to avoid None for single element
    std = float(std_val) if std_val is not None else 0.0
    fig.add_trace(go.Scatter(x=lx, y=mean + 3.29 * std / np.sqrt(lx), mode="lines", **tw))
    fig.add_trace(go.Scatter(x=lx, y=mean - 3.29 * std / np.sqrt(lx), mode="lines", **tw))
    fig.add_hline(y=mean, line_width=1.5, line=line_option)
    fig.update_yaxes(range=_get_lim(float(df[y].min()), float(df[y].max())))
    fig.update_xaxes(range=_get_lim(0, nm))
    return fig


class Subplot:
    """Helper class for creating matplotlib subplots with sequential access.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from statract.figure import Subplot
        >>>
        >>> # Create 2x2 subplot grid
        >>> subplot = Subplot(2, 2, figsize=(12, 8))
        >>>
        >>> # Get axes sequentially
        >>> ax1 = subplot.get_ax()  # Top-left
        >>> ax1.plot([1, 2, 3], [1, 4, 9])
        >>>
        >>> ax2 = subplot.get_ax()  # Top-right
        >>> ax2.bar(['A', 'B', 'C'], [1, 2, 3])
        >>>
        >>> ax3 = subplot.get_ax()  # Bottom-left
        >>> ax3.scatter([1, 2, 3], [2, 3, 4])
        >>>
        >>> ax4 = subplot.get_ax()  # Bottom-right
        >>> ax4.hist([1, 2, 2, 3, 3, 3])
        >>>
        >>> # Display the figure
        >>> subplot.show()
        >>>
        >>> # Single plot
        >>> subplot = Subplot(1, 1)
        >>> ax = subplot.get_ax()
        >>> ax.plot([1, 2, 3])
        >>> subplot.show()
    """

    def __init__(self, nrows: int, ncols: int, figsize: tuple[float, float] = (10, 5)) -> None:
        """Initialize subplot grid.

        Args:
            nrows: Number of rows in the grid
            ncols: Number of columns in the grid
            figsize: Figure size tuple (width, height) in inches
        """
        self.ncols = ncols
        self.nrows = nrows
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        self.fig.set_layout_engine("tight")
        self.counter = 0

    def get_ax(self) -> plt.Axes:
        """Get the next axis in the grid.

        Returns:
            Matplotlib axis object for the next position in the grid

        Raises:
            AssertionError: If all axes have been used

        Examples:
            >>> from statract.figure import Subplot
            >>> import matplotlib.pyplot as plt
            >>>
            >>> subplot = Subplot(2, 2)
            >>> ax = subplot.get_ax()
            >>> ax.plot([1, 2, 3], [1, 4, 9])
            >>> subplot.show()
        """
        assert self.counter < self.ncols * self.nrows
        if self.ncols == 1 and self.nrows == 1:
            ret = self.axes
        elif self.ncols == 1 or self.nrows == 1:
            ret = self.axes[self.counter]
        else:
            ret = self.axes[self.counter // self.ncols][self.counter % self.ncols]
        self.counter += 1
        return ret

    def show(self) -> None:
        """Display the figure with tight layout.

        Examples:
            >>> from statract.figure import Subplot
            >>> import matplotlib.pyplot as plt
            >>>
            >>> subplot = Subplot(1, 1)
            >>> ax = subplot.get_ax()
            >>> ax.plot([1, 2, 3])
            >>> subplot.show()  # Displays the figure
        """
        plt.tight_layout()
        plt.show()


def labeled_subplot_grid(
    labels: list[str],
    *,
    ncols: int | None = None,
    cell_size: tuple[float, float] = (5.0, 4.0),
) -> tuple[Any | None, list[Any]]:
    """Return ``(fig, axes)`` with one subplot per label. Empty labels → ``(None, [])``."""
    n = len(labels)
    if n == 0:
        return None, []
    ncol = ncols or min(n, 3)
    nrows = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(
        nrows,
        ncol,
        figsize=(cell_size[0] * ncol, cell_size[1] * nrows),
        squeeze=False,
    )
    return fig, [axes.flat[i] for i in range(n)]


__all__ = [
    "FigureWidthPreset",
    "ForestRow",
    "ForestTextCol",
    "Subplot",
    "concat_images",
    "concat_plotly_figures",
    "funnel_plot",
    "labeled_subplot_grid",
    "parallel_categories_keeporder",
    "rows_from_prepared",
    "rows_from_prepared_grouped",
    "rows_from_prepared_named_groups",
    "save_forest_panel_rows",
    "save_prepared_hr_forest",
]
