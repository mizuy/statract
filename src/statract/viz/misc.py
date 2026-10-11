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
    """Arrange images in a grid with ``ncols`` columns on a white background.

    Args:
        images: List of PIL Images to concatenate
        ncols: Number of columns in the grid (default: 10)

    Returns:
        PIL Image (RGB) with all images arranged in a grid

    Raises:
        ValueError: If images list is empty
    """
    if not images:
        raise ValueError("Cannot concatenate empty list of images")
    ncols = max(1, min(int(ncols), len(images)))
    nrows = -(-len(images) // ncols)
    col_widths = [0] * ncols
    row_heights = [0] * nrows
    for idx, img in enumerate(images):
        width, height = img.size
        row, col = divmod(idx, ncols)
        col_widths[col] = max(col_widths[col], int(width))
        row_heights[row] = max(row_heights[row], int(height))
    col_offsets = np.concatenate(([0], np.cumsum(col_widths))).astype(int)
    row_offsets = np.concatenate(([0], np.cumsum(row_heights))).astype(int)
    canvas = Image.new("RGB", (int(col_offsets[-1]), int(row_offsets[-1])), (255, 255, 255))
    for idx, img in enumerate(images):
        row, col = divmod(idx, ncols)
        canvas.paste(img, (int(col_offsets[col]), int(row_offsets[row])))
    return canvas

def _get_lim(p: float, q: float) -> tuple[float, float]:
    """Calculate axis limits with small margin.

    Args:
        p: Minimum value
        q: Maximum value

    Returns:
        Tuple of (min_limit, max_limit) with 0.5% margin

    Examples:
        >>> from statract.viz.misc import get_lim
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
        >>> from statract.viz.misc import funnel_plot
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
