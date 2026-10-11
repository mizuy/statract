"""Expand pandoc citations (``[@key]``) with pandoc citeproc.

Python-Markdown renders the pages, so pandoc never sees the whole page. This
hook runs pandoc only on the citations: it collects each ``[@key; @key2]``
group on a page, renders the groups and the bibliography in one pandoc call,
and puts the HTML back into the Markdown. Admonitions, math and other MkDocs
syntax stay untouched.

The bibliography is ``references.bib`` at the repo root. It holds only the
entries the docs cite, copied from Paperpile by bibliome::

    cd ~/lab/tools/bibliome
    task bib PAPER_DIR=~/lab/statract

The reference list goes where the page has a ``::: {#refs}`` / ``:::`` pair,
or at the end of the page. Works listed but not cited in the text go in the
page front matter as ``nocite: "[@key1; @key2]"``.

Pages without citations do not need pandoc.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("mkdocs.plugins.citations")

_ROOT = Path(__file__).resolve().parent.parent
BIB = _ROOT / "references.bib"
CSL = _ROOT / "tools" / "csl" / "vancouver.csl"

# Same shape bibliase scans: a bracket that starts with ``@key``. A bracket
# followed by ``(`` or ``[`` is a link, not a citation.
CITE_RE = re.compile(r"\[(@[A-Za-z][\w:-]*[^\[\]\n]*)\](?![(\[])")
# Fenced code blocks and inline code spans are never scanned.
CODE_RE = re.compile(r"^(```|~~~).*?^\1[ \t]*$|`[^`\n]+`", re.M | re.S)
REFS_RE = re.compile(r"^:::\s*\{#refs\}[ \t]*\n:::[ \t]*$", re.M)

_CITE_DIV_RE = re.compile(r'<div id="cite-(\d+)">\s*<p>(.*?)</p>\s*</div>', re.S)


def _citations(markdown: str) -> list[re.Match]:
    code = [m.span() for m in CODE_RE.finditer(markdown)]
    return [
        m
        for m in CITE_RE.finditer(markdown)
        if not any(a <= m.start() < b for a, b in code)
    ]


def _pandoc_input(groups: list[str], nocite: str) -> str:
    meta = "---\nlink-citations: true\n"
    if nocite:
        meta += f"nocite: |\n  {nocite}\n"
    body = "".join(f"::: {{#cite-{i}}}\n[{g}]\n:::\n\n" for i, g in enumerate(groups))
    return f"{meta}---\n\n{body}::: {{#refs}}\n:::\n"


def render(groups: list[str], *, nocite: str = "", bib: Path = BIB, csl: Path = CSL):
    """Rendered HTML for each citation group, and the bibliography HTML."""
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise RuntimeError("the docs cite papers; install pandoc to build them")
    if not bib.is_file():
        raise FileNotFoundError(f"{bib} is missing; build it with `task bib` in bibliome")
    proc = subprocess.run(
        [
            pandoc,
            "--from=markdown",
            "--to=html",
            "--wrap=none",
            "--citeproc",
            f"--bibliography={bib}",
            f"--csl={csl}",
        ],
        input=_pandoc_input(groups, nocite),
        capture_output=True,
        text=True,
        check=True,
    )
    for line in proc.stderr.splitlines():
        log.warning("pandoc: %s", line)
    cites = {int(i): html for i, html in _CITE_DIV_RE.findall(proc.stdout)}
    start = proc.stdout.find('<div id="refs"')
    refs = proc.stdout[start:].strip() if start >= 0 else ""
    return [cites[i] for i in range(len(groups))], refs


def expand(markdown: str, *, nocite: str = "", bib: Path = BIB, csl: Path = CSL) -> str:
    """Markdown with citations and the reference list rendered as HTML."""
    matches = _citations(markdown)
    if not matches and not nocite:
        return markdown
    groups = [m.group(1) for m in matches]
    cites, refs = render(groups, nocite=nocite, bib=bib, csl=csl)
    out, pos = [], 0
    for m, html in zip(matches, cites):
        out += [markdown[pos : m.start()], html]
        pos = m.end()
    out.append(markdown[pos:])
    text = "".join(out)
    block = f"\n{refs}\n"
    if REFS_RE.search(text):
        return REFS_RE.sub(lambda _: block, text, count=1)
    return f"{text.rstrip()}\n\n{block}"


def on_page_markdown(markdown: str, *, page, config, files) -> str:
    nocite = page.meta.get("nocite") or ""
    return expand(markdown, nocite=str(nocite).strip())
