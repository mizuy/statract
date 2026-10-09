from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from statract.figure import concat_images, funnel_plot


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
