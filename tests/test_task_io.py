"""Task output helpers moved from endolab."""

from __future__ import annotations

import polars as pl

from statract.task_io import prepare_task_output, save_frames


def test_prepare_task_output_writes_csv(tmp_path) -> None:
    script = tmp_path / "run.py"
    script.write_text("", encoding="utf-8")
    out = prepare_task_output(script)
    assert out == tmp_path / "out"
    assert out.is_dir()
    frame = pl.DataFrame({"a": [1]})
    path = out / "a.csv"
    save_frames({"a": path}, {"a": frame})
    assert pl.read_csv(path)["a"].to_list() == [1]
