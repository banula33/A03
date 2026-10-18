"""Export all notebook text, code, outputs and plots without requiring LaTeX."""
import base64
import ast
import io
import re
import textwrap
from html import escape
from pathlib import Path
import nbformat
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, Image

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CodeSmall", fontName="Courier", fontSize=8.2,
                          leading=10.2, spaceAfter=6, backColor=colors.HexColor("#f4f4f4")))
styles["BodyText"].fontSize = 10
styles["BodyText"].leading = 14
story = []

def inline(text):
    text = escape(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    return text

def markdown(text):
    for block in re.split(r"\n\s*\n", text.strip()):
        if not block.strip():
            continue
        if block.startswith("#"):
            heading, _, body = block.partition("\n")
            level = len(heading) - len(heading.lstrip("#"))
            story.append(Paragraph(inline(heading.lstrip("# ")), styles[f"Heading{min(level, 3)}"]))
            if body:
                story.append(Paragraph(inline(body).replace("\n", " "), styles["BodyText"]))
        else:
            text = inline(block).replace("  \n", "<br/>").replace("\n", " ")
            story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 5))

def preformatted(text):
    lines = []
    for line in text.splitlines():
        lines.extend(textwrap.wrap(line.expandtabs(4), width=100,
                                  replace_whitespace=False, drop_whitespace=False) or [""])
    # Short blocks let ReportLab break long cells across pages.
    for i in range(0, len(lines), 38):
        story.append(Preformatted("\n".join(lines[i:i + 38]), styles["CodeSmall"]))

def experiment_code(source):
    # Discussion is included as rendered prose, rather than duplicated as strings.
    lines = source.splitlines()
    excluded = set()
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "display"):
            excluded.update(range(node.lineno - 1, node.end_lineno))
    return "\n".join(line for i, line in enumerate(lines) if i not in excluded)

notebook = nbformat.read("230070T_A03.ipynb", as_version=4)
for cell in notebook.cells:
    if cell.cell_type == "markdown":
        markdown(cell.source)
    elif cell.cell_type == "code":
        preformatted(experiment_code(cell.source))
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                # Download progress bars are not experiment results.
                lines = [line for line in output.text.splitlines()
                         if not ("it/s" in line or "B/s" in line or "Downloading:" in line)]
                if lines:
                    preformatted("\n".join(lines))
            elif output.output_type in ("display_data", "execute_result"):
                data = output.data
                if "image/png" in data:
                    raw = io.BytesIO(base64.b64decode(data["image/png"]))
                    width, height = ImageReader(raw).getSize()
                    scale = min(480 / width, 470 / height)
                    raw.seek(0)
                    story.append(Image(raw, width=width * scale, height=height * scale))
                elif "text/markdown" in data:
                    markdown(data["text/markdown"])
                elif "text/plain" in data:
                    preformatted(data["text/plain"])
            elif output.output_type == "error":
                raise RuntimeError("Notebook contains an execution error")
        story.append(Spacer(1, 8))

def page_number(canvas, document):
    canvas.setFont("Helvetica", 8)
    canvas.drawString(42, 25, "EN3160 Assignment 3 | 230070T")
    canvas.drawRightString(553, 25, str(document.page))

pdf = Path("230070T_a02.pdf")
SimpleDocTemplate(str(pdf), pagesize=(595.28, 841.89), rightMargin=42,
                  leftMargin=42, topMargin=38, bottomMargin=40,
                  title="EN3160 Assignment 3 - 230070T").build(
                      story, onFirstPage=page_number, onLaterPages=page_number)
print("Exported", pdf)
