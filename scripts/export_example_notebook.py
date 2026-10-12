#!/usr/bin/env python3
"""Render saved notebook outputs as a self-contained HTML page, without execution.

Requires nbformat, Markdown and Pygments. Example:
    python scripts/export_example_notebook.py examples/h01_single_neuron.ipynb
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import markdown
import nbformat
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


CSS = """
:root { color-scheme: light; font: 17px/1.6 system-ui, sans-serif; color: #192532; }
body { margin: 0; background: #f3f5f7; }
main { max-width: 1080px; margin: 2rem auto; padding: 2rem 3rem; background: white; }
a { color: #155ba0; text-underline-offset: .16em; }
h1, h2, h3 { line-height: 1.25; color: #102b42; }
h1 { font-size: 2rem; } h2 { margin-top: 2rem; font-size: 1.4rem; }
header { padding-bottom: 1rem; border-bottom: 1px solid #dce3e8; }
.saved { margin: 0; color: #4d6172; font-size: .9rem; }
.fingerprint { overflow-wrap: anywhere; font-size: .75rem; color: #617383; }
.cell { margin: 1.3rem 0; }
.cell-label { margin: 0 0 .3rem; color: #617383; font-size: .75rem; }
.code, .output pre { padding: 1rem; overflow-x: auto; background: #f5f7fa; }
pre, code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
pre { margin: 0; font-size: .8rem; line-height: 1.5; }
p code, li code { background: #f0f3f6; padding: .1rem .25rem; font-size: .9em;
                 overflow-wrap: anywhere; }
.output { margin: .75rem 0; overflow-x: auto; }
.output img { display: block; max-width: 100%; height: auto; margin: 1rem auto; }
table { border-collapse: collapse; font-size: .88rem; max-width: 100%; }
th, td { padding: .45rem .6rem; border-bottom: 1px solid #dce3e8; vertical-align: top; }
thead { background: #f0f4f7; } tbody tr:nth-child(even) { background: #fafbfd; }
blockquote { margin-left: 0; padding-left: 1rem; border-left: 3px solid #c3d6e7; }
footer { margin-top: 2rem; border-top: 1px solid #dce3e8; padding-top: 1rem; }
@media (max-width: 720px) { main { margin: 0; padding: 1.25rem; } h1 { font-size: 1.6rem; } }
@media print {
  body { background: white; } main { max-width: none; margin: 0; padding: 0; }
  .output { overflow: visible; } pre { white-space: pre-wrap; overflow-wrap: anywhere; }
  h1, h2, h3 { break-after: avoid; } img, table { break-inside: avoid; }
}
"""


def text_value(value: str | list[str]) -> str:
    return value if isinstance(value, str) else "".join(value)


class OfflineHTML(HTMLParser):
    """Keep relative links usable after export and reject active/remote assets."""

    def __init__(self, notebook_directory: Path, output_directory: Path):
        super().__init__(convert_charrefs=False)
        self.notebook_directory = notebook_directory
        self.output_directory = output_directory
        self.parts: list[str] = []
        self.style_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "iframe", "object", "embed", "form", "link"}:
            raise ValueError(f"unsupported active HTML element: {tag}")
        rewritten = []
        for name, value in attrs:
            if name.lower().startswith("on") or name in {"srcset", "action"}:
                raise ValueError(f"unsupported HTML attribute: {name}")
            if name == "src" and not (
                tag == "img" and value and value.startswith("data:image/png;base64,")
            ):
                raise ValueError("HTML assets must be embedded PNG images")
            if name == "style" and value:
                self.check_css(value)
            if name == "href" and value:
                parts = urlsplit(value)
                if parts.scheme.lower() not in {"", "http", "https", "mailto"}:
                    raise ValueError(f"unsupported hyperlink scheme: {parts.scheme}")
                if not parts.scheme and not parts.netloc and parts.path:
                    if parts.path.startswith("/"):
                        raise ValueError("absolute filesystem hyperlinks are unsupported")
                    target = self.notebook_directory / unquote(parts.path)
                    relative = os.path.relpath(target, self.output_directory)
                    value = urlunsplit(("", "", quote(relative, safe="/"), parts.query, parts.fragment))
            rewritten.append(name if value is None else f'{name}="{html.escape(value, quote=True)}"')
        self.parts.append("<" + tag + (" " + " ".join(rewritten) if rewritten else "") + ">")
        if tag == "style":
            self.style_depth += 1

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        self.parts.append(f"</{tag}>")
        if tag == "style":
            self.style_depth -= 1

    @staticmethod
    def check_css(value):
        if re.search(r"@import|url\s*\(|expression\s*\(", value, re.IGNORECASE):
            raise ValueError("external or active CSS is unsupported")

    def handle_data(self, data):
        if self.style_depth:
            self.check_css(data)
        self.parts.append(data)

    def handle_entityref(self, name):
        self.parts.append(f"&{name};")

    def handle_charref(self, name):
        self.parts.append(f"&#{name};")

    def handle_comment(self, data):
        self.parts.append(f"<!--{data}-->")


def render_notebook(notebook_path: Path, output_path: Path) -> tuple[str, dict]:
    source = notebook_path.read_bytes()
    notebook = nbformat.reads(source.decode("utf-8"), as_version=4)
    nbformat.validate(notebook)
    formatter = HtmlFormatter(cssclass="code")
    blocks, output_receipts = [], []
    title = notebook_path.stem
    for cell_index, cell in enumerate(notebook.cells):
        cell_source = cell.source
        if cell.cell_type == "markdown":
            if not blocks:
                match = re.search(r"^#\s+(.+)$", cell_source, re.MULTILINE)
                if match:
                    title = match.group(1)
            body = markdown.markdown(cell_source, extensions=["tables", "fenced_code"])
        elif cell.cell_type == "code":
            count = cell.execution_count
            label = f"Code · In [{count if count is not None else ' '}]:"
            body = f'<p class="cell-label">{html.escape(label)}</p>'
            body += highlight(cell_source, PythonLexer(), formatter)
            for output_index, output in enumerate(cell.outputs):
                data = output.get("data", {})
                receipt = {"cell_index": cell_index, "output_index": output_index}
                if "image/png" in data:
                    encoded = text_value(data["image/png"])
                    payload = base64.b64decode(encoded, validate=True)
                    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
                        raise ValueError("image/png output does not contain a PNG image")
                    rendered = '<img alt="Saved notebook figure" src="data:image/png;base64,' + encoded + '">'
                    receipt.update(mime="image/png", sha256=hashlib.sha256(payload).hexdigest())
                elif "text/html" in data:
                    rendered = text_value(data["text/html"])
                    receipt.update(mime="text/html", sha256=hashlib.sha256(rendered.encode()).hexdigest())
                elif "text/plain" in data:
                    rendered = "<pre>" + html.escape(text_value(data["text/plain"])) + "</pre>"
                    receipt["mime"] = "text/plain"
                elif output.output_type == "stream":
                    rendered = "<pre>" + html.escape(output.text) + "</pre>"
                    receipt["mime"] = "stream"
                else:
                    raise ValueError(f"unsupported notebook output at cell {cell_index}: {output.output_type}")
                body += '<div class="output">' + rendered + "</div>"
                output_receipts.append(receipt)
        else:
            raise ValueError(f"unsupported notebook cell type: {cell.cell_type}")
        parser = OfflineHTML(notebook_path.parent.resolve(), output_path.parent.resolve())
        parser.feed(body)
        parser.close()
        blocks.append(f'<section class="cell" id="cell-{cell_index}">' + "".join(parser.parts) + "</section>")
    digest = hashlib.sha256(source).hexdigest()
    notebook_link = quote(os.path.relpath(notebook_path.resolve(), output_path.parent.resolve()), safe="/")
    header = ('<header><p class="saved">Saved execution · '
              f'<a href="{html.escape(notebook_link, quote=True)}">Open the canonical notebook</a></p>'
              '<p class="saved">Code, numerical tables and figures are saved notebook content. '
              'Use the notebook to reproduce the processing run.</p>'
              f'<p class="fingerprint">Notebook SHA256: {digest}</p></header>')
    document = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                f'<title>{html.escape(title)}</title>\n<style>\n{CSS}\n'
                f'{formatter.get_style_defs(".code")}\n</style>\n</head>\n<body><main>\n'
                + header + "\n" + "\n".join(blocks)
                + '\n<footer class="saved">Self-contained HTML rendering of saved notebook outputs.</footer>'
                + '\n</main></body>\n</html>\n')
    receipt = {"status": "passed", "notebook_sha256": digest,
               "html_sha256": hashlib.sha256(document.encode()).hexdigest(),
               "cell_count": len(notebook.cells), "outputs": output_receipts,
               "execution_performed": False, "external_assets": [], "scripts": []}
    return document, receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, help="default: notebook path with .html suffix")
    parser.add_argument("--receipt", type=Path, help="optional JSON integrity receipt")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or args.notebook.with_suffix(".html")
    try:
        if output.resolve() == args.notebook.resolve():
            raise ValueError("HTML output must be different from the input notebook")
        if args.receipt and args.receipt.resolve() in {output.resolve(), args.notebook.resolve()}:
            raise ValueError("receipt path must be different from the notebook and HTML output")
        for path in [output, args.receipt]:
            if path and path.exists() and not args.overwrite:
                raise ValueError(f"output already exists: {path}; pass --overwrite to replace it")
        document, receipt = render_notebook(args.notebook, output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(document, encoding="utf-8")
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {output} ({len(document.encode()):,} bytes; no notebook execution)")
    except (OSError, ValueError, nbformat.ValidationError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
