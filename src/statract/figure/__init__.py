"""Small figure helpers: a Plotly funnel plot and image grids.

Statistical figures are ``plot_forest``, ``plot_survival`` and ``plot_tree``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from PIL import Image


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


__all__ = [
    "concat_images",
    "funnel_plot",
]
