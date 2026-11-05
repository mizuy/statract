"""R source literals for scalars passed to ``assign`` (no R needed)."""

from __future__ import annotations

import pytest

from statract.r import _escape_r_string, _r_scalar_literal


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('a"b', 'a\\"b'),
        ("a\\b", "a\\\\b"),
        ("a'b", "a'b"),
        ('a\\"b', 'a\\\\\\"b'),
        ("a\nb\tc\rd", "a\\nb\\tc\\rd"),
    ],
)
def test_escape_r_string(value: str, expected: str) -> None:
    assert _escape_r_string(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (42, "42L"),
        (-3, "-3L"),
        (2**31, "2147483648.0"),
        (True, "TRUE"),
        (False, "FALSE"),
        (1.5, "1.5"),
        (0.1, "0.1"),
        (float("nan"), "NaN"),
        (float("inf"), "Inf"),
        (float("-inf"), "-Inf"),
        ('a"b', '"a\\"b"'),
    ],
)
def test_r_scalar_literal(value: object, expected: str) -> None:
    assert _r_scalar_literal(value) == expected
