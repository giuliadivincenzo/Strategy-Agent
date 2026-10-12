"""
Document parsing (Phase 2: text + images).

Turns a PDF or DOCX file into (text, warnings, images):
- `text` is plain text with page/section markers injected inline, so the LLM
  (and the schema's `source_reference` field) can cite where a claim came from.
- `warnings` is a list of human-readable strings describing anything the
  parser could not handle cleanly (used to fill `processing_metadata` and to
  nudge the LLM's confidence down on affected content).
- `images` is a list of dicts, one per extracted image:
  {"location": "page 2", "mime_type": "image/png", "base64_data": "..."}.
  Only pages/documents already flagged in `warnings` as image-heavy or
  text-empty produce an entry here -- this is a deliberate hybrid design:
  cheap text extraction stays the default path for everything else, and the
  (slower, more expensive) image path is only used where the text pipeline
  has already signaled it can't fully cope. See agent.py/prompts.py for how
  these get sent to a vision-capable model.
"""

import base64
from io import BytesIO
from pathlib import Path

import pdfplumber
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph


class UnsupportedFileType(Exception):
    pass


def extract_text(file_path: str) -> tuple[str, list[str], list[dict]]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext == ".docx":
        return _extract_docx(file_path)
    else:
        raise UnsupportedFileType(f"Unsupported file type: {ext}. Only .pdf and .docx are supported.")


def _extract_pdf(file_path: str) -> tuple[str, list[str], list[dict]]:
    warnings: list[str] = []
    pages_text: list[str] = []
    images: list[dict] = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""

            if not page_text.strip():
                warnings.append(
                    f"Page {i}: no extractable text found (likely a scanned "
                    f"image, a full-page chart, or a blank page). Content was skipped."
                )
                pages_text.append(f"[--- PAGE {i} (no extractable text; possibly image/scan) ---]")
                _append_page_screenshot(page, i, images, warnings)
                continue

            # Attempt table extraction separately -- pdfplumber's plain text
            # extraction often mangles table layout, so we pull tables out
            # explicitly and append them as a labeled block.
            table_blocks = []
            try:
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables, start=1):
                    rows = [
                        " | ".join(cell if cell else "" for cell in row)
                        for row in table
                        if row
                    ]
                    if rows:
                        table_blocks.append(
                            f"[Table {t_idx} on page {i}]\n" + "\n".join(rows)
                        )
            except Exception as e:
                warnings.append(f"Page {i}: table extraction failed ({e}).")

            block = f"[--- PAGE {i} ---]\n{page_text}"
            if table_blocks:
                block += "\n\n" + "\n\n".join(table_blocks)
            pages_text.append(block)

            # Heuristic: flag pages that likely contain a chart/diagram the
            # text layer won't capture (has embedded images, low text density).
            try:
                if len(page.images) > 0:
                    warnings.append(
                        f"Page {i}: contains {len(page.images)} embedded image(s) "
                        f"(possible chart/diagram/photo). A screenshot of this "
                        f"page was extracted for visual analysis."
                    )
                    _append_page_screenshot(page, i, images, warnings)
            except Exception:
                pass

    return "\n\n".join(pages_text), warnings, images


def _append_page_screenshot(page, page_number: int, images: list[dict], warnings: list[str]) -> None:
    """
    Renders a full screenshot of a flagged PDF page (not just the embedded
    image object) and appends it to `images` as base64. A full-page
    screenshot is used rather than trying to crop out just the embedded
    image, since pdfplumber's raw image objects are often just one layer of
    a composited chart (background, axes, data points may be separate
    objects) -- the rendered page is what a human would actually see.
    """
    try:
        rendered = page.to_image(resolution=150)
        buffer = BytesIO()
        rendered.original.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        images.append(
            {
                "location": f"page {page_number}",
                "mime_type": "image/png",
                "base64_data": b64,
            }
        )
    except Exception as e:
        warnings.append(f"Page {page_number}: could not render page screenshot for visual analysis ({e}).")


def _extract_docx(file_path: str) -> tuple[str, list[str], list[dict]]:
    warnings: list[str] = []
    doc = DocxDocument(file_path)
    blocks: list[str] = []
    images: list[dict] = []
    current_heading = None
    table_count = 0

    # Iterate the body in document order so tables land where they belong
    # relative to surrounding paragraphs, rather than being extracted separately.
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            if para.style and para.style.name and para.style.name.lower().startswith("heading"):
                current_heading = text
                blocks.append(f"[--- SECTION: {text} ---]")
            else:
                blocks.append(text)
        elif child.tag.endswith("}tbl"):
            table_count += 1
            table = Table(child, doc)
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(" | ".join(cells))
            label = f" (section: {current_heading})" if current_heading else ""
            blocks.append(f"[Table {table_count}{label}]\n" + "\n".join(rows))

    # Extract embedded images directly from the document's internal parts
    # (rather than just counting them, as before). Each image relationship
    # holds the raw picture bytes, which we base64-encode for the API.
    try:
        image_rels = [rel for rel in doc.part.rels.values() if "image" in rel.reltype]
        for idx, rel in enumerate(image_rels, start=1):
            try:
                blob = rel.target_part.blob
                content_type = getattr(rel.target_part, "content_type", "image/png")
                b64 = base64.b64encode(blob).decode("ascii")
                location = f"embedded image {idx}"
                if current_heading:
                    location += f" (near section: {current_heading})"
                images.append(
                    {
                        "location": location,
                        "mime_type": content_type,
                        "base64_data": b64,
                    }
                )
            except Exception as e:
                warnings.append(f"Could not extract embedded image {idx} ({e}).")
        if image_rels:
            warnings.append(
                f"Document contains {len(image_rels)} embedded image(s) "
                f"(possible chart/diagram). Extracted for visual analysis."
            )
    except Exception:
        pass

    if not blocks:
        warnings.append("No extractable text found in document body.")

    return "\n\n".join(blocks), warnings, images
