"""Load Mermaid from the site itself, before the Material bundle.

Material for MkDocs renders ``pre.mermaid`` and, if ``window.mermaid`` is
missing, injects ``https://unpkg.com/mermaid``. That CDN request is what
leaves flowcharts as source text on GitHub Pages. A synchronous script in
``head`` defines the global first, so the bundle never calls unpkg.
"""

from mkdocs.utils import get_relative_url

_MERMAID = "javascripts/vendor/mermaid.min.js"


def on_post_page(output: str, *, page, config) -> str:
    if 'class="mermaid"' not in output:
        return output
    src = get_relative_url(_MERMAID, page.url)
    tag = f'<script src="{src}"></script>'
    return output.replace("</head>", f"{tag}\n</head>", 1)
