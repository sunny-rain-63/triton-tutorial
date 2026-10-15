#!/usr/bin/env python3
"""Rebuild ex1-vector_add-en/vector_add.ipynb and vector_add.html from blog.md.

The English chapter is authored in Markdown (`ex1-vector_add-en/blog.md`); the
notebook and the rendered HTML are generated artifacts, so they never drift
from the prose.

Usage:
    python tools/build_en_docs.py

Requires only the `markdown` package (for the HTML export).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "ex1-vector_add-en" / "blog.md"
NB_OUT = ROOT / "ex1-vector_add-en" / "vector_add.ipynb"
HTML_OUT = ROOT / "ex1-vector_add-en" / "vector_add.html"

FENCE = re.compile(r"^```(\w*)\s*$")


def split_cells(markdown_text: str):
    """Split the chapter into (cell_type, source) pairs.

    Fenced blocks tagged ```Python / ```python become code cells; every other
    fence (```text, ```bash output samples) stays inside markdown.
    """
    cells: list[tuple[str, str]] = []
    buffer: list[str] = []
    code: list[str] | None = None
    in_other_fence = False

    def flush_markdown() -> None:
        text = "".join(buffer).strip("\n")
        buffer.clear()
        if text:
            cells.append(("markdown", text))

    for line in markdown_text.splitlines(keepends=True):
        fence = FENCE.match(line.rstrip("\n"))

        if in_other_fence:
            buffer.append(line)
            if line.rstrip("\n") == "```":
                in_other_fence = False
            continue

        if code is not None:
            if line.rstrip("\n") == "```":
                cells.append(("code", "".join(code).strip("\n")))
                code = None
            else:
                code.append(line)
            continue

        if fence:
            lang = fence.group(1).lower()
            if lang in ("python", "py"):
                flush_markdown()
                code = []
            else:
                buffer.append(line)
                in_other_fence = True
            continue

        buffer.append(line)

    flush_markdown()
    return cells


def build_notebook(cells) -> dict:
    nb_cells = []
    for cell_type, source in cells:
        lines = source.splitlines(keepends=True)
        if cell_type == "code":
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines,
            })
        else:
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": lines,
            })
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vector Addition on Ascend NPU &middot; Hands-On Triton Tutorial</title>
<style>
  :root {{
    --bg: #ffffff; --fg: #1f2328; --muted: #59636e;
    --border: #d1d9e0; --code-bg: #f6f8fa; --accent: #0969da;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1;
      --border: #3d444d; --code-bg: #161b22; --accent: #4493f8;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1;
    --border: #3d444d; --code-bg: #161b22; --accent: #4493f8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--fg); margin: 0;
    font: 16px/1.7 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans",
          Helvetica, Arial, "Apple Color Emoji", sans-serif;
  }}
  main {{ max-width: 860px; margin: 0 auto; padding: 48px 16px 96px; }}
  h1, h2, h3, h4 {{ line-height: 1.3; margin-top: 2em; }}
  h2 {{ padding-bottom: .3em; border-bottom: 1px solid var(--border); }}
  a {{ color: var(--accent); }}
  code {{
    background: var(--code-bg); padding: .2em .4em; border-radius: 6px;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
    font-size: .9em;
  }}
  pre {{
    background: var(--code-bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; overflow-x: auto;
  }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{
    margin: 1em 0; padding: .3em 1em; color: var(--muted);
    border-left: 4px solid var(--border);
  }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; display: block;
           overflow-x: auto; }}
  th, td {{ border: 1px solid var(--border); padding: 6px 13px; text-align: left; }}
  th {{ background: var(--code-bg); }}
  .note {{
    border: 1px solid var(--border); border-left: 4px solid var(--accent);
    background: var(--code-bg); border-radius: 6px; padding: 12px 16px;
    margin-bottom: 32px; font-size: .95em;
  }}
</style>
</head>
<body>
<main>
<div class="note">
Generated from <code>ex1-vector_add-en/blog.md</code> by
<code>tools/build_en_docs.py</code>. Do not edit this file directly &mdash;
edit the Markdown source and re-run the script.
</div>
{body}
</main>
</body>
</html>
"""


def build_html(markdown_text: str) -> str:
    import markdown as md

    body = md.markdown(
        markdown_text,
        extensions=["fenced_code", "tables", "toc", "sane_lists"],
    )
    return HTML_TEMPLATE.format(body=body)


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    cells = split_cells(text)

    n_code = sum(1 for t, _ in cells if t == "code")
    NB_OUT.write_text(
        json.dumps(build_notebook(cells), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    HTML_OUT.write_text(build_html(text), encoding="utf-8")

    print(f"{NB_OUT.relative_to(ROOT)}: {len(cells)} cells ({n_code} code)")
    print(f"{HTML_OUT.relative_to(ROOT)}: {HTML_OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
