"""Tests for confusion_matrix and related functions."""

import polars as pl
import pytest

from statract import confusion_matrix, ratio


class TestRatio:
    """Test ratio function."""

    def test_ratio_basic(self):
        """Test basic ratio calculation."""
        result = ratio(3, 10)
        assert result == "30.0% 3/10"
        assert "30.0%" in result
        assert "3/10" in result

    def test_ratio_25_percent(self):
        """Test ratio with 25%."""
        result = ratio(1, 4)
        assert result == "25.0% 1/4"

    def test_ratio_50_percent(self):
        """Test ratio with 50%."""
        result = ratio(5, 10)
        assert result == "50.0% 5/10"

    def test_ratio_zero(self):
        """Test ratio with zero numerator."""
        result = ratio(0, 10)
        assert result == "0.0% 0/10"

    def test_ratio_100_percent(self):
        """Test ratio with 100%."""
        result = ratio(10, 10)
        assert result == "100.0% 10/10"


class TestConfusionMatrix:
    """Test confusion_matrix class."""

    def test_basic_initialization(self):
        """Test basic confusion matrix initialization."""
        cm = confusion_matrix(TP=80, TN=90, FP=10, FN=20)
        assert cm.TP == 80
        assert cm.TN == 90
        assert cm.FP == 10
        assert cm.FN == 20
        assert cm.n == 200

    def test_metrics_calculation(self):
        """Test that metrics are calculated correctly."""
        cm = confusion_matrix(TP=80, TN=90, FP=10, FN=20)

        # Sensitivity (TPR, Recall, Se)
        expected_se = 80 / (80 + 20)  # TP / (TP + FN)
        assert cm.Se == expected_se
        assert cm.Recall == expected_se
        assert cm.TPR == expected_se

        # Specificity (TNR, Sp)
        expected_sp = 90 / (90 + 10)  # TN / (TN + FP)
        assert cm.Sp == expected_sp
        assert cm.TNR == expected_sp

        # Precision (PPV)
        expected_ppv = 80 / (80 + 10)  # TP / (TP + FP)
        assert cm.Precision == expected_ppv
        assert cm.PPV == expected_ppv

        # NPV
        expected_npv = 90 / (90 + 20)  # TN / (TN + FN)
        assert cm.NPV == expected_npv

        # Accuracy
        expected_acc = (80 + 90) / 200
        assert cm.Acc == expected_acc

    def test_derived_metrics(self):
        """Test derived metrics (FNR, FPR, FDR, FOR)."""
        cm = confusion_matrix(TP=80, TN=90, FP=10, FN=20)

        # False Negative Rate
        assert cm.FNR == 1 - cm.TPR
        # False Positive Rate
        assert cm.FPR == 1 - cm.TNR
        # False Discovery Rate
        assert cm.FDR == 1 - cm.PPV
        # False Omission Rate
        assert cm.FOR == 1 - cm.NPV

    def test_f1_score(self):
        """Test F1 score calculation."""
        cm = confusion_matrix(TP=80, TN=90, FP=10, FN=20)
        expected_f1 = 2 * 80 / (2 * 80 + 10 + 20)
        assert cm.F1 == expected_f1

    def test_likelihood_ratios(self):
        """Test likelihood ratio calculations."""
        cm = confusion_matrix(TP=80, TN=90, FP=10, FN=20)

        # Positive likelihood ratio
        expected_lrp = cm.Se / (1 - cm.Sp)
        assert cm.LRp == expected_lrp

        # Negative likelihood ratio
        expected_lrn = cm.Sp / (1 - cm.Se)
        assert cm.LRn == expected_lrn

        # Diagnostic odds ratio
        assert cm.DOR == cm.LRp / cm.LRn

    def test_prevalence(self):
        """Test prevalence calculation."""
        cm = confusion_matrix(TP=80, TN=90, FP=10, FN=20)
        expected_prev = (80 + 10) / 200  # (TP + FP) / n
        assert cm.Prev == expected_prev

    def test_summary(self, capsys):
        """Test summary method output."""
        cm = confusion_matrix(TP=80, TN=90, FP=10, FN=20)
        cm.summary()
        captured = capsys.readouterr()
        assert "TP" in captured.out
        assert "TN" in captured.out
        assert "FP" in captured.out
        assert "FN" in captured.out
        assert "PPV" in captured.out
        assert "NPV" in captured.out
        assert "Se" in captured.out
        assert "Sp" in captured.out

    def test_from_expr(self):
        """Test from_expr class method."""
        df = pl.DataFrame(
            {
                "pred": [True, True, False, False, True, False],
                "truth": [True, False, True, False, True, True],
            }
        )

        cm = confusion_matrix.from_expr(pl.col("pred"), pl.col("truth"), df)

        # Calculate expected values
        # TP: pred=True, truth=True -> 2 cases (indices 0, 4)
        # TN: pred=False, truth=False -> 1 case (index 3)
        # FP: pred=True, truth=False -> 1 case (index 1)
        # FN: pred=False, truth=True -> 2 cases (indices 2, 5)
        assert cm.TP == 2
        assert cm.TN == 1
        assert cm.FP == 1
        assert cm.FN == 2
        assert cm.n == 6

    def test_from_expr_all_true(self):
        """Test from_expr with all true predictions."""
        df = pl.DataFrame(
            {
                "pred": [True, True, True],
                "truth": [True, True, True],
            }
        )

        # This will cause ZeroDivisionError in confusion_matrix.__init__
        # because TNR = TN / (TN + FP) = 0 / (0 + 0)
        # This is an edge case that the implementation doesn't handle
        with pytest.raises(ZeroDivisionError):
            cm = confusion_matrix.from_expr(pl.col("pred"), pl.col("truth"), df)

    def test_from_expr_all_false(self):
        """Test from_expr with all false predictions."""
        df = pl.DataFrame(
            {
                "pred": [False, False, False],
                "truth": [False, False, False],
            }
        )

        # This will cause ZeroDivisionError in confusion_matrix.__init__
        # because TPR = TP / (TP + FN) = 0 / (0 + 0)
        # This is an edge case that the implementation doesn't handle
        with pytest.raises(ZeroDivisionError):
            cm = confusion_matrix.from_expr(pl.col("pred"), pl.col("truth"), df)

    def test_from_expr_perfect_prediction(self):
        """Test from_expr with perfect predictions."""
        df = pl.DataFrame(
            {
                "pred": [True, True, False, False],
                "truth": [True, True, False, False],
            }
        )

        # This will cause ZeroDivisionError in confusion_matrix.__init__
        # because LRp = Se / (1 - Sp) = 1 / (1 - 1) = 1 / 0
        # This is an edge case that the implementation doesn't handle
        with pytest.raises(ZeroDivisionError):
            cm = confusion_matrix.from_expr(pl.col("pred"), pl.col("truth"), df)

    def test_from_expr_all_wrong(self):
        """Test from_expr with all wrong predictions."""
        df = pl.DataFrame(
            {
                "pred": [True, True, False, False],
                "truth": [False, False, True, True],
            }
        )

        # This will cause ZeroDivisionError in confusion_matrix.__init__
        # because DOR = LRp / LRn where LRn = Sp / (1 - Se) = 0 / (1 - 0) = 0 / 1 = 0
        # and then DOR = LRp / 0
        # This is an edge case that the implementation doesn't handle
        with pytest.raises(ZeroDivisionError):
            cm = confusion_matrix.from_expr(pl.col("pred"), pl.col("truth"), df)
