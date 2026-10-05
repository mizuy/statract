import io
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from PIL import Image

from statract.figure import (
    Subplot,
    concat_images,
    concat_plotly_figures,
    funnel_plot,
    parallel_categories_keeporder,
)


class TestParallelCategoriesKeeporder:
    def test_basic_functionality(self):
        # Create test data with Polars DataFrame
        df = pl.DataFrame(
            {
                "cat1": ["A", "B", "A", "C", "B"],
                "cat2": ["X", "Y", "X", "Z", "Y"],
                "cat3": ["P", "Q", "P", "R", "Q"],
            }
        )

        dimensions = ["cat1", "cat2", "cat3"]
        orders = [["C", "B", "A"], ["Z", "Y", "X"], ["R", "Q", "P"]]

        with patch("statract.figure.px") as mock_px:
            mock_fig = MagicMock()
            mock_fig.update_traces = MagicMock(return_value=mock_fig)
            mock_px.parallel_categories = MagicMock(return_value=mock_fig)

            result = parallel_categories_keeporder(df, dimensions, orders)

            # Should call px.parallel_categories with Polars DataFrame
            mock_px.parallel_categories.assert_called_once()
            call_args = mock_px.parallel_categories.call_args
            assert isinstance(call_args[0][0], pl.DataFrame)

            # Should call update_traces with proper options
            mock_fig.update_traces.assert_called_once()

            # Verify the options structure
            call_args = mock_fig.update_traces.call_args[1]
            assert "dimensions" in call_args
            assert len(call_args["dimensions"]) == 3

            # Each dimension should have categoryorder and categoryarray
            for option in call_args["dimensions"]:
                assert "categoryorder" in option
                assert "categoryarray" in option
                assert option["categoryorder"] == "array"

    def test_dimension_order_mismatch(self):
        df = pl.DataFrame({"cat1": ["A", "B"], "cat2": ["X", "Y"]})
        dimensions = ["cat1", "cat2"]
        orders = [["A", "B"]]  # Only one order for two dimensions

        with pytest.raises(AssertionError):
            parallel_categories_keeporder(df, dimensions, orders)

    def test_order_with_missing_categories(self):
        df = pl.DataFrame(
            {
                "cat1": ["A", "B", "C"],
                "cat2": ["X", "Y", "Z"],
            }
        )

        dimensions = ["cat1", "cat2"]
        orders = [["A", "D"], ["X"]]  # 'D' not in data, 'Y' and 'Z' not in order

        with patch("statract.figure.px") as mock_px:
            mock_fig = MagicMock()
            mock_fig.update_traces = MagicMock(return_value=mock_fig)
            mock_px.parallel_categories = MagicMock(return_value=mock_fig)

            result = parallel_categories_keeporder(df, dimensions, orders)

            # Should still work and include missing categories
            call_args = mock_fig.update_traces.call_args[1]
            dimensions_options = call_args["dimensions"]

            # First dimension should have 'A' first, then 'B', 'C'
            assert "A" in dimensions_options[0]["categoryarray"]
            assert dimensions_options[0]["categoryarray"][0] == "A"


class TestConcatPlotlyFigures:
    @patch("statract.figure.pio")
    @patch("statract.figure.Image")
    def test_concat_plotly_figures(self, mock_image_class, mock_pio):
        # Mock the figures
        fig0 = MagicMock()
        fig1 = MagicMock()

        # Mock PIL Images
        mock_img0 = MagicMock()
        mock_img0.width = 100
        mock_img0.height = 200

        mock_img1 = MagicMock()
        mock_img1.width = 150
        mock_img1.height = 180

        # Mock Image.open to return our mock images
        mock_image_class.open.side_effect = [mock_img0, mock_img1]

        # Mock the new combined image
        mock_combined = MagicMock()
        mock_image_class.new.return_value = mock_combined

        result = concat_plotly_figures(fig0, fig1)

        # Should write both figures to PNG
        assert mock_pio.write_image.call_count == 2

        # Should open both images
        assert mock_image_class.open.call_count == 2

        # Should create new image with combined width and max height
        mock_image_class.new.assert_called_once_with(
            "RGB",
            (250, 200),  # 100 + 150 width, max(200, 180) height
        )

        # Should paste both images
        assert mock_combined.paste.call_count == 2

        assert result == mock_combined


class TestConcatImages:
    @patch("statract.figure.Image")
    def test_concat_images_basic(self, mock_image_class):
        # Create mock images
        images = []
        for i in range(3):
            mock_img = MagicMock()
            mock_img.size = (100 + i * 10, 200 + i * 5)  # Different sizes
            images.append(mock_img)

        # Mock the new combined image
        mock_combined = MagicMock()
        mock_image_class.new.return_value = mock_combined

        result = concat_images(images, ncols=3)  # Use ncols >= n_images

        # Should create new image
        mock_image_class.new.assert_called_once()

        # Should paste all images
        assert mock_combined.paste.call_count == 3

        assert result == mock_combined

    @patch("statract.figure.Image")
    def test_concat_images_single_row(self, mock_image_class):
        # Create mock images for single row
        images = []
        for i in range(3):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            images.append(mock_img)

        mock_combined = MagicMock()
        mock_image_class.new.return_value = mock_combined

        result = concat_images(images, ncols=10)  # More columns than images

        # Should still work
        mock_image_class.new.assert_called_once()
        assert mock_combined.paste.call_count == 3
        assert result == mock_combined

    @patch("statract.figure.Image")
    def test_concat_images_empty_list(self, mock_image_class):
        # The code should raise ValueError for empty list
        with pytest.raises(ValueError, match="Cannot concatenate empty list"):
            concat_images([], ncols=2)


class TestFunnelPlot:
    @patch("statract.figure.px")
    @patch("statract.figure.go")
    def test_funnel_plot_basic(self, mock_go, mock_px):
        # Create test data with Polars DataFrame
        df = pl.DataFrame(
            {
                "n_count": [10, 20, 30, 40, 50],
                "y_value": [0.1, 0.2, 0.15, 0.25, 0.18],
            }
        )
        raw_y = pl.Series([0.1, 0.2, 0.15, 0.25, 0.18, 0.22, 0.19])

        # Mock plotly objects
        mock_fig = MagicMock()
        mock_px.scatter.return_value = mock_fig
        mock_scatter = MagicMock()
        mock_go.Scatter.return_value = mock_scatter

        result = funnel_plot(df, "n_count", "y_value", raw_y)

        # Should create scatter plot with Polars DataFrame
        mock_px.scatter.assert_called_once()
        call_args = mock_px.scatter.call_args
        assert isinstance(call_args[0][0], pl.DataFrame)

        # Should add traces for confidence limits
        assert mock_fig.add_trace.call_count == 2

        # Should add horizontal line for mean
        mock_fig.add_hline.assert_called_once()

        # Should update axes
        mock_fig.update_yaxes.assert_called_once()
        mock_fig.update_xaxes.assert_called_once()

        assert result == mock_fig

    @patch("statract.figure.px")
    @patch("statract.figure.go")
    def test_funnel_plot_with_kwargs(self, mock_go, mock_px):
        df = pl.DataFrame(
            {
                "n_count": [10, 20, 30],
                "y_value": [0.1, 0.2, 0.15],
                "category": ["A", "B", "A"],
            }
        )
        raw_y = pl.Series([0.1, 0.2, 0.15])

        mock_fig = MagicMock()
        mock_px.scatter.return_value = mock_fig
        mock_scatter = MagicMock()
        mock_go.Scatter.return_value = mock_scatter

        result = funnel_plot(df, "n_count", "y_value", raw_y, color="category")

        # Should pass kwargs to scatter plot
        call_args = mock_px.scatter.call_args
        assert "color" in call_args[1]
        assert call_args[1]["color"] == "category"

        assert result == mock_fig


class TestSubplot:
    @patch("statract.figure.plt")
    def test_subplot_single_plot(self, mock_plt):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        subplot = Subplot(1, 1, figsize=(8, 6))

        # Should create subplot with correct parameters
        mock_plt.subplots.assert_called_once_with(1, 1, figsize=(8, 6))

        # Should set layout engine
        mock_fig.set_layout_engine.assert_called_once_with("tight")

        # Get axis should return the single axis
        ax = subplot.get_ax()
        assert ax == mock_ax
        assert subplot.counter == 1

    @patch("statract.figure.plt")
    def test_subplot_multiple_plots_1d(self, mock_plt):
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        subplot = Subplot(1, 3, figsize=(12, 4))

        # Counter starts at 0, uses counter for indexing, then increments
        ax1 = subplot.get_ax()  # counter=0, uses axes[0]
        ax2 = subplot.get_ax()  # counter=1, uses axes[1]
        ax3 = subplot.get_ax()  # counter=2, uses axes[2]

        assert ax1 == mock_axes[0]
        assert ax2 == mock_axes[1]
        assert ax3 == mock_axes[2]
        assert subplot.counter == 3

    @patch("statract.figure.plt")
    def test_subplot_multiple_plots_2d(self, mock_plt):
        mock_fig = MagicMock()
        mock_axes = [[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        subplot = Subplot(2, 2, figsize=(10, 8))

        # Get axes in order
        ax1 = subplot.get_ax()  # Should be [0][0]
        ax2 = subplot.get_ax()  # Should be [0][1]
        ax3 = subplot.get_ax()  # Should be [1][0]
        ax4 = subplot.get_ax()  # Should be [1][1]

        assert ax1 == mock_axes[0][0]
        assert ax2 == mock_axes[0][1]
        assert ax3 == mock_axes[1][0]
        assert ax4 == mock_axes[1][1]
        assert subplot.counter == 4

    @patch("statract.figure.plt")
    def test_subplot_too_many_axes(self, mock_plt):
        mock_fig = MagicMock()
        mock_axes = [MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        subplot = Subplot(1, 1)

        # Get the one available axis
        subplot.get_ax()

        # Should raise assertion error when trying to get more
        with pytest.raises(AssertionError):
            subplot.get_ax()

    @patch("statract.figure.plt")
    def test_subplot_show(self, mock_plt):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        subplot = Subplot(1, 1)
        subplot.show()

        # Should call tight_layout and show
        mock_plt.tight_layout.assert_called_once()
        mock_plt.show.assert_called_once()


class TestUtilityFunctions:
    def test_get_lim_via_funnel_plot(self):
        """Test _get_lim indirectly through funnel_plot."""
        from statract.figure import funnel_plot
        import polars as pl

        # Create test data
        df = pl.DataFrame({"n": [10, 20, 30, 40, 50], "mean": [0.5, 0.52, 0.48, 0.51, 0.49]})
        raw_y = pl.Series([0.4, 0.5, 0.6, 0.45, 0.55, 0.5, 0.52, 0.48])

        # funnel_plot uses _get_lim internally
        fig = funnel_plot(df, n="n", y="mean", raw_y=raw_y)
        assert fig is not None


class TestEdgeCases:
    @patch("statract.figure.px")
    def test_parallel_categories_empty_dataframe(self, mock_px):
        df = pl.DataFrame()
        dimensions = []
        orders = []

        mock_fig = MagicMock()
        mock_fig.update_traces = MagicMock(return_value=mock_fig)
        mock_px.parallel_categories = MagicMock(return_value=mock_fig)

        result = parallel_categories_keeporder(df, dimensions, orders)

        # Should still work with empty data
        mock_px.parallel_categories.assert_called_once()
        assert result == mock_fig

    @patch("statract.figure.px")
    @patch("statract.figure.go")
    def test_funnel_plot_single_point(self, mock_go, mock_px):
        df = pl.DataFrame(
            {
                "n_count": [10],
                "y_value": [0.1],
            }
        )
        raw_y = pl.Series([0.1])

        mock_fig = MagicMock()
        mock_px.scatter.return_value = mock_fig
        mock_scatter = MagicMock()
        mock_go.Scatter.return_value = mock_scatter

        result = funnel_plot(df, "n_count", "y_value", raw_y)

        # Should still work with single point
        mock_px.scatter.assert_called_once()
        assert result == mock_fig
