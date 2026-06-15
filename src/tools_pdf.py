import re
from pathlib import Path

from langchain_core.tools import tool
from markdown import markdown
from xhtml2pdf import pisa

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def _safe_filename(name: str) -> str:
    """Clean filename so it is safe for filesystem usage."""
    value = name.strip().lower().replace(" ", "-")
    value = re.sub(r"[^a-z0-9\\-_.]", "", value)
    return value or "raport"


@tool
def save_markdown_as_pdf(markdown_text: str, filename: str) -> str:
    """Save markdown report as PDF in reports directory."""
    try:
        safe_name = _safe_filename(filename)
        md_path = REPORTS_DIR / f"{safe_name}.md"
        html_path = REPORTS_DIR / f"{safe_name}.html"
        pdf_path = REPORTS_DIR / f"{safe_name}.pdf"

        md_path.write_text(markdown_text, encoding="utf-8")

        html_body = markdown(markdown_text, output_format="html5")
        html_full = f"""
        <html>
          <head>
            <meta charset="utf-8" />
            <style>
              body {{ font-family: Arial, sans-serif; margin: 30px; line-height: 1.5; }}
              h1, h2, h3 {{ color: #222; }}
              code {{ background: #f4f4f4; padding: 2px 4px; }}
            </style>
          </head>
          <body>
            {html_body}
          </body>
        </html>
        """
        html_path.write_text(html_full, encoding="utf-8")

        with pdf_path.open("wb") as out_pdf:
            result = pisa.CreatePDF(html_full, dest=out_pdf)

        if result.err:
            return "Błąd: nie udało się utworzyć PDF."
        return str(pdf_path.resolve())
    except Exception as exc:  # pragma: no cover - filesystem path
        return f"Błąd podczas zapisu PDF: {exc}"
