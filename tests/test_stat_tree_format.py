import numpy as np
import polars as pl

from statract import conditional_tree


def test_format_and_terminal_count():
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=n)
    g = rng.choice(["a", "b", "c"], n)
    y = (rng.random(n) < 0.2 + 0.5 * (x > 0.5) + 0.2 * (g == "c")).astype(float)
    tree = conditional_tree(pl.DataFrame({"y": y, "x": x, "g": g}), "y", ["x", "g"])
    text = tree.format()
    lines = text.splitlines()
    assert lines[0] == "[1] root"
    leaves = [line for line in lines if "(n = " in line]
    assert len(leaves) == tree.n_terminal() >= 2
    assert sum(int(line.split("(n = ")[1].rstrip(")")) for line in leaves) == n
    assert any("x <= " in line for line in lines)
    assert any("g in {" in line for line in lines)


def test_format_single_node():
    tree = conditional_tree(pl.DataFrame({"y": [0.0, 1.0, 0.0], "x": [1.0, 2.0, 3.0]}), "y", ["x"])
    assert tree.format() == "[1] root: 0.333 (n = 3)\n"
    assert tree.n_terminal() == 1
