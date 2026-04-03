"""
Document parsing for PDF and HTML files.
Uses PyMuPDF for PDFs and BeautifulSoup for HTML.
Docling is attempted first for PDFs if available.
"""

import logging
from pathlib import Path

from bs4 import BeautifulSoup

from src.utils import detect_language, make_doc_id, normalize_whitespace

log = logging.getLogger(__name__)

try:
    from docling.document_converter import DocumentConverter
    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False
    log.info("docling not available — using PyMuPDF fallback for PDFs")

try:
    import fitz
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False
    log.warning("PyMuPDF (fitz) not available — PDF parsing disabled")


def _detect_lang_from_url(url: str) -> str | None:
    if "/en/" in url or url.endswith("/en"):
        return "en"
    if "/sl/" in url or url.endswith("/sl"):
        return "sl"
    return None


def _parse_pdf_docling(path: Path) -> list[dict]:
    from docling.datamodel.base_models import InputFormat
    from docling_core.types.doc import DocItemLabel

    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc = result.document

    HEADING_LABELS = {
        DocItemLabel.TITLE,
        DocItemLabel.SECTION_HEADER,
    }

    records = []
    current_section = "main"

    for item, _ in doc.iterate_items():
        if not hasattr(item, "text"):
            continue
        text = normalize_whitespace(item.text)
        if not text:
            continue

        label = getattr(item, "label", None)
        if label in HEADING_LABELS:
            current_section = text[:120]
            continue

        records.append({
            "section": current_section,
            "text": text,
        })

    return records


def _parse_pdf_pymupdf(path: Path) -> list[dict]:
    doc = fitz.open(str(path))
    records = []
    for page_num, page in enumerate(doc):
        text = normalize_whitespace(page.get_text())
        if not text:
            continue
        records.append({
            "section": f"page_{page_num + 1}",
            "text": text,
        })
    doc.close()
    return records


def parse_pdf(path: Path) -> dict | None:
    doc_id = make_doc_id(str(path))
    title = path.stem.replace("_", " ").replace("-", " ")

    if _DOCLING_AVAILABLE:
        try:
            sections = _parse_pdf_docling(path)
            source = "docling"
        except Exception as e:
            log.warning("docling failed for %s: %s — falling back to PyMuPDF", path.name, e)
            sections = []

    if not _DOCLING_AVAILABLE or not sections:
        if not _FITZ_AVAILABLE:
            log.error("No PDF parser available for %s", path.name)
            return None
        try:
            sections = _parse_pdf_pymupdf(path)
            source = "pymupdf"
        except Exception as e:
            log.error("PyMuPDF failed for %s: %s", path.name, e)
            return None

    if not sections:
        log.warning("No text extracted from PDF: %s", path.name)
        return None

    full_text = "\n\n".join(s["text"] for s in sections)
    language = detect_language(full_text)

    return {
        "doc_id": doc_id,
        "source_path": str(path),
        "title": title,
        "sections": sections,
        "text": full_text,
        "url": "",
        "language": language,
        "parser": source,
        "metadata": {"file_type": "pdf", "num_sections": len(sections)},
    }


def _extract_html_title(soup: BeautifulSoup) -> str:
    title_div = soup.find("div", class_="left-container-text")
    if title_div:
        title_el = title_div.find("div", class_="title")
        if title_el:
            return title_el.get_text(strip=True)

    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)

    return "Unknown"


def _extract_html_sections(soup: BeautifulSoup) -> list[dict]:
    content_div = (
        soup.find("div", class_="body-page-rows")
        or soup.find("div", class_="novica-content")
        or soup.find("div", class_="field-items")
        or soup.find("article")
        or soup.find("main")
    )

    if content_div is None:
        body = soup.find("body")
        content_div = body if body else soup

    for tag in content_div.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    sections = []
    current_section = "main"
    current_parts: list[str] = []

    for el in content_div.children:
        if not hasattr(el, "name") or el.name is None:
            continue

        if el.name in ("h1", "h2", "h3", "h4"):
            if current_parts:
                text = normalize_whitespace(" ".join(current_parts))
                if text:
                    sections.append({"section": current_section, "text": text})
                current_parts = []
            current_section = el.get_text(strip=True) or current_section
        else:
            text = el.get_text(separator=" ", strip=True)
            if text:
                current_parts.append(text)

    if current_parts:
        text = normalize_whitespace(" ".join(current_parts))
        if text:
            sections.append({"section": current_section, "text": text})

    return sections


def parse_html(path: Path) -> dict | None:
    doc_id = make_doc_id(str(path))

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log.error("Could not read HTML file %s: %s", path.name, e)
        return None

    soup = BeautifulSoup(raw, "html.parser")
    title = _extract_html_title(soup)
    sections = _extract_html_sections(soup)

    if not sections:
        log.warning("No content extracted from HTML: %s", path.name)
        return None

    full_text = "\n\n".join(s["text"] for s in sections)
    url_tag = soup.find("link", rel="canonical")
    url = url_tag["href"] if url_tag and url_tag.get("href") else ""

    lang_from_url = _detect_lang_from_url(url)
    html_tag = soup.find("html")
    lang_attr = str(html_tag.get("lang", "")).lower() if html_tag else ""
    if lang_from_url:
        language = lang_from_url
    elif lang_attr.startswith("en"):
        language = "en"
    else:
        language = detect_language(full_text)

    return {
        "doc_id": doc_id,
        "source_path": str(path),
        "title": title,
        "sections": sections,
        "text": full_text,
        "url": url,
        "language": language,
        "parser": "beautifulsoup",
        "metadata": {"file_type": "html", "num_sections": len(sections)},
    }


def parse_text(path: Path) -> dict | None:
    doc_id = make_doc_id(str(path))
    try:
        text = normalize_whitespace(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        log.error("Could not read file %s: %s", path.name, e)
        return None
    if not text:
        log.warning("Empty file: %s", path.name)
        return None
    return {
        "doc_id": doc_id,
        "source_path": str(path),
        "title": path.stem,
        "sections": [{"section": "main", "text": text}],
        "text": text,
        "url": "",
        "language": detect_language(text),
        "parser": "plaintext",
        "metadata": {"file_type": path.suffix.lstrip(".")},
    }


def parse_file(path: Path) -> dict | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return parse_pdf(path)
        elif suffix in (".html", ".htm"):
            return parse_html(path)
        elif suffix in (".txt", ".md"):
            return parse_text(path)
        else:
            log.warning("Unsupported file type: %s", path.name)
            return None
    except Exception as e:
        log.error("Unexpected error parsing %s: %s", path.name, e)
        return None


def parse_directory(raw_dir: Path) -> list[dict]:
    extensions = {".pdf", ".html", ".htm", ".txt", ".md"}
    files = [f for f in raw_dir.rglob("*") if f.suffix.lower() in extensions]
    log.info("Found %d files to parse in %s", len(files), raw_dir)

    documents = []
    for path in files:
        doc = parse_file(path)
        if doc:
            documents.append(doc)
            log.info("Parsed [%s] %s", doc["language"], path.name)
        else:
            log.warning("Skipped: %s", path.name)

    log.info("Successfully parsed %d / %d files", len(documents), len(files))
    return documents
