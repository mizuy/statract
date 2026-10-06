"""Load Mermaid from the site itself, before the Material bundle.

Material for MkDocs renders ``pre.mermaid`` and, if ``window.mermaid`` is
missing, injects ``https://unpkg.com/mermaid``. That CDN request is what
leaves flowcharts as source text on GitHub Pages. A synchronous script in
``head`` defines the global first, so the bundle never calls unpkg.
"""

import os
from pathlib import Path

from mkdocs.utils import get_relative_url

_MERMAID = "javascripts/vendor/mermaid.min.js"


# Old published paths. Directory URLs, so each source page is ``<stem>/index.html``.
_REDIRECTS = {
    "stat/overview.md": "models/overview.md",
    "stat/quickstart.md": "models/quickstart.md",
    "stat/formula.md": "models/formula.md",
    "stat/vs-r.md": "models/vs-r.md",
    "stat/benchmarks.md": "models/benchmarks.md",
    "stat/examples.md": "examples/index.md",
    "stat/example-gallery-policy.md": "examples/policy.md",
    "stat/benchmark-plan.md": "dev/benchmark-plan.md",
    "stat/speed-plan.md": "dev/speed-plan.md",
    "stat/r-parity-plan.md": "dev/r-parity-plan.md",
    "stat/r-lazy-plan.md": "dev/r-lazy-plan.md",
    "stat/examples/surv_colon.md": "examples/surv_colon.md",
    "stat/examples/logit_indo.md": "examples/logit_indo.md",
    "stat/examples/psm_rhc.md": "examples/psm_rhc.md",
    "stat/examples/cif_pbc.md": "examples/cif_pbc.md",
    "stat/examples/aft_rotterdam.md": "examples/aft_rotterdam.md",
    "stat/examples/cox_retinopathy.md": "examples/cox_retinopathy.md",
    "stat/examples/lmm_pbcseq.md": "examples/lmm_pbcseq.md",
    "stat/examples/iptw_nhefs.md": "examples/iptw_nhefs.md",
    "stat/examples/pred_support.md": "examples/pred_support.md",
    "stat/examples/cea_sicksicker.md": "examples/cea_sicksicker.md",
    "api/stat/tableone.md": "api/tableone/tableone.md",
    "api/stat/agg.md": "api/tableone/agg.md",
    "api/stat/stat.md": "api/tableone/stat.md",
    "api/stat/forest.md": "api/viz/forest.md",
    "api/stat/survival.md": "api/viz/survival.md",
    "api/stat/formula.md": "api/models/formula.md",
    "api/stat/fit.md": "api/models/fit.md",
    "api/stat/covariance.md": "api/models/covariance.md",
    "api/stat/linear_tests.md": "api/models/linear_tests.md",
    "api/stat/surv.md": "api/models/surv.md",
    "api/stat/matching.md": "api/models/matching.md",
    "api/stat/gam.md": "api/models/gam.md",
    "api/stat/tree.md": "api/models/tree.md",
    "api/stat/mixed.md": "api/models/mixed.md",
    "api/stat/regression.md": "api/models/regression.md",
    "api/stat/confusion.md": "api/models/confusion.md",
    "api/stat/r.md": "api/models/r.md",
}


def _redirect_html(href: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="ja"><head><meta charset="utf-8">\n'
        f'<meta http-equiv="refresh" content="0; url={href}">\n'
        f'<link rel="canonical" href="{href}">\n'
        f'<script>location.replace("{href}")</script>\n'
        "</head>\n"
        f'<body><a href="{href}">Moved</a></body></html>\n'
    )


def on_post_build(config) -> None:
    """Write directory-URL redirects for the old ``stat/`` docs paths."""
    site = Path(config["site_dir"])
    for src, dst in _REDIRECTS.items():
        page = site / Path(src).with_suffix("") / "index.html"
        target = site / Path(dst).with_suffix("")
        href = Path(os.path.relpath(target, page.parent)).as_posix() + "/"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(_redirect_html(href), encoding="utf-8")


def on_post_page(output: str, *, page, config) -> str:
    if 'class="mermaid"' not in output:
        return output
    src = get_relative_url(_MERMAID, page.url)
    tag = f'<script src="{src}"></script>'
    return output.replace("</head>", f"{tag}\n</head>", 1)
