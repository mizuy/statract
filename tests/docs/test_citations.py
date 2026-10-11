"""The MkDocs hook that expands ``[@key]`` citations with pandoc."""

import importlib.util
import shutil
from pathlib import Path

import pytest

_HOOK = Path(__file__).resolve().parents[2] / "tools" / "mkdocs_citations.py"
_spec = importlib.util.spec_from_file_location("mkdocs_citations", _HOOK)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

BIB = """\
@ARTICLE{Austin2011-sk,
  title        = {An introduction to propensity score methods},
  author       = {Austin, Peter C},
  journaltitle = {Multivariate behavioral research},
  volume       = {46},
  pages        = {399--424},
  date         = {2011-05}
}

@ARTICLE{Stuart2010-ke,
  title        = {Matching methods for causal inference},
  author       = {Stuart, Elizabeth A},
  journaltitle = {Statistical science},
  volume       = {25},
  pages        = {1--21},
  date         = {2010}
}
"""

PAGE = """\
!!! note
    Review [@Stuart2010-ke]. Both [@Austin2011-sk; @Stuart2010-ke].

`[@Austin2011-sk]` and [@Austin2011-sk](https://example.org) stay.

```python
x = "[@Stuart2010-ke]"
```

## 文献

::: {#refs}
:::

end
"""


def test_finds_citations_outside_code_and_links():
    assert [m.group(1) for m in mc._citations(PAGE)] == [
        "@Stuart2010-ke",
        "@Austin2011-sk; @Stuart2010-ke",
    ]


def test_page_without_citations_is_unchanged():
    assert mc.expand("no [citations] here, mail a@b.org") == "no [citations] here, mail a@b.org"


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_expand_numbers_in_citation_order(tmp_path):
    bib = tmp_path / "references.bib"
    bib.write_text(BIB, encoding="utf-8")
    out = mc.expand(PAGE, bib=bib)
    # Stuart is cited first, so it is reference 1.
    assert '<a href="#ref-Stuart2010-ke" role="doc-biblioref">1</a>' in out
    assert '<a href="#ref-Austin2011-sk" role="doc-biblioref">2</a>' in out
    assert "Austin PC." in out
    # The list replaces the marker, before the text that follows it.
    assert "::: {#refs}" not in out
    assert out.index('<div id="refs"') < out.index("end")
    # Code and links are left alone.
    assert 'x = "[@Stuart2010-ke]"' in out
    assert "[@Austin2011-sk](https://example.org)" in out


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_nocite_lists_uncited_work(tmp_path):
    bib = tmp_path / "references.bib"
    bib.write_text(BIB, encoding="utf-8")
    out = mc.expand("Text [@Stuart2010-ke].\n", nocite="[@Austin2011-sk]", bib=bib)
    assert 'id="ref-Austin2011-sk"' in out
    assert out.rstrip().endswith("</div>")
