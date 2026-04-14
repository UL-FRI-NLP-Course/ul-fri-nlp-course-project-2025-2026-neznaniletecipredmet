"""Document parsing for PDF, HTML and DOCX files.

PDFs:
- If Docling is installed, it is attempted first. For scanned PDFs, Docling may
    run OCR (depending on its available OCR backend) and return extracted text.
- Otherwise, PyMuPDF is used as a fallback.

Images:
- Standalone images are not OCR'd by this project. They are supported only via
    a required sidecar text/markdown file.
"""

import importlib.util
import json
import logging
import os
import subprocess
import shutil
from pathlib import Path

from bs4 import BeautifulSoup

import config
from src.utils import (
    detect_language,
    make_stable_doc_id,
    normalize_whitespace,
    sha256_file,
)

log = logging.getLogger(__name__)

_DOCX_AVAILABLE = importlib.util.find_spec("docx") is not None
if not _DOCX_AVAILABLE:
    log.info("python-docx not available - DOCX parsing disabled")

_DOCLING_AVAILABLE = importlib.util.find_spec("docling.document_converter") is not None
if not _DOCLING_AVAILABLE:
    log.info("docling not available - using PyMuPDF fallback for PDFs")

_FITZ_AVAILABLE = importlib.util.find_spec("fitz") is not None
if not _FITZ_AVAILABLE:
    log.warning("PyMuPDF (fitz) not available - PDF parsing disabled")

_DOCLING_CONVERTER = None
_DOCLING_CONVERTER_OCR_MODE: str | None = None


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _get_docling_converter():
    """Create (and cache) a Docling DocumentConverter with our preferred OCR settings.

    Preference:
    - If CUDA is available: let Docling auto-select the best OCR backend.
    - If CUDA is not available: force Tesseract CLI OCR to avoid heavy OCR stacks on CPU.
    """

    global _DOCLING_CONVERTER, _DOCLING_CONVERTER_OCR_MODE

    want_mode = "gpu" if _torch_cuda_available() else "cpu"
    if _DOCLING_CONVERTER is not None and _DOCLING_CONVERTER_OCR_MODE == want_mode:
        return _DOCLING_CONVERTER

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        OcrAutoOptions,
        PdfPipelineOptions,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption


    def _tesseract_candidates(cmd: str) -> list[str]:
        candidates: list[str] = []

        # If user provided a full/relative path, try it directly.
        if cmd:
            candidates.append(cmd)
            # Common: user provides 'tesseract' but executable is 'tesseract.exe'.
            if os.name == "nt" and not cmd.lower().endswith(".exe"):
                candidates.append(f"{cmd}.exe")

        # Common install locations (UB Mannheim / winget).
        if os.name == "nt":
            for env_var in ("ProgramFiles", "ProgramFiles(x86)"):
                base = os.environ.get(env_var)
                if base:
                    candidates.append(str(Path(base) / "Tesseract-OCR" / "tesseract.exe"))
            local_app_data = os.environ.get("LOCALAPPDATA")
            if local_app_data:
                candidates.append(str(Path(local_app_data) / "Programs" / "Tesseract-OCR" / "tesseract.exe"))

        return candidates


    def _clean_exe_path(value: str) -> str | None:
        if not value:
            return None
        v = value.strip().strip('"')
        # Some registry entries contain extra args; keep only up to .exe.
        lower = v.lower()
        if ".exe" in lower:
            i = lower.index(".exe") + 4
            v = v[:i]
        return v


    def _try_registry_app_path() -> str | None:
        if os.name != "nt":
            return None
        try:
            import winreg
        except Exception:
            return None

        key_names = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\tesseract.exe",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths\tesseract.exe",
        ]
        roots = [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]

        for root in roots:
            for key_name in key_names:
                try:
                    with winreg.OpenKey(root, key_name) as k:
                        value, _ = winreg.QueryValueEx(k, "")
                    cleaned = _clean_exe_path(str(value))
                    if cleaned and Path(cleaned).exists():
                        return cleaned
                except Exception:
                    continue
        return None


    def _is_usable_tesseract(exe: str) -> bool:
        try:
            proc = subprocess.run(
                [exe, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            return proc.returncode == 0
        except Exception:
            return False


    def _find_tesseract(cmd: str) -> str | None:
        # 1) PATH lookup.
        found = shutil.which(cmd) or (shutil.which(f"{cmd}.exe") if os.name == "nt" else None)
        if found and _is_usable_tesseract(found):
            return found

        # 2) Registry App Paths (can make an exe runnable even when 'where.exe' doesn't find it).
        reg = _try_registry_app_path()
        if reg and _is_usable_tesseract(reg):
            return reg

        # 3) Common install locations or explicit path.
        for cand in _tesseract_candidates(cmd):
            p = _clean_exe_path(cand)
            if not p:
                continue
            if Path(p).exists() and _is_usable_tesseract(p):
                return p

        return None

    if want_mode == "cpu":
        tesseract_exe = _find_tesseract(config.TESSERACT_CMD)
        if tesseract_exe is None:
            log.warning(
                "Tesseract not found/usable (cmd=%s). Falling back to Docling OCR auto.",
                config.TESSERACT_CMD,
            )
            ocr_options = OcrAutoOptions()
        else:
            log.info("Using Tesseract OCR (cpu): %s", tesseract_exe)
            ocr_options = TesseractCliOcrOptions(
                lang=list(config.DOCLING_OCR_LANGS_TESSERACT),
                tesseract_cmd=tesseract_exe,
                path=config.TESSERACT_DATA_PATH,
            )
    else:
        ocr_options = OcrAutoOptions()

    # Docling's StandardPdfPipeline renders and caches page images during the
    # 'preprocess' stage. Large/scanned PDFs can otherwise trigger native
    # 'std::bad_alloc' (out-of-memory) errors on some pages.
    pdf_options = PdfPipelineOptions(
        ocr_options=ocr_options,
        ocr_batch_size=1,
        layout_batch_size=1,
        table_batch_size=1,
        queue_max_size=8,
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        },
    )

    _DOCLING_CONVERTER = converter
    _DOCLING_CONVERTER_OCR_MODE = want_mode
    return _DOCLING_CONVERTER


def _detect_lang_from_url(url: str) -> str | None:
    if "/en/" in url or url.endswith("/en"):
        return "en"
    if "/sl/" in url or url.endswith("/sl"):
        return "sl"
    return None


def _parse_pdf_docling(path: Path) -> list[dict]:
    from docling_core.types.doc import DocItemLabel

    converter = _get_docling_converter()
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
    import fitz

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
    sha = sha256_file(path)
    doc_id = make_stable_doc_id(sha256=sha, fallback=str(path))
    title = path.stem.replace("_", " ").replace("-", " ")

    sections: list[dict] = []
    source = ""

    if _DOCLING_AVAILABLE:
        try:
            sections = _parse_pdf_docling(path)
            source = "docling"
        except Exception as e:
            log.warning("docling failed for %s: %s - falling back to PyMuPDF", path.name, e)
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
        "parser": source or "unknown",
        "metadata": {"file_type": "pdf", "num_sections": len(sections), "sha256": sha},
    }


def _iter_docx_blocks(doc):
    """Yield Paragraph and Table objects in document order."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    # python-docx doesn't provide a public ordered iterator over mixed
    # paragraphs/tables, but we can walk the underlying XML body.
    for child in doc.element.body.iterchildren():
        tag = getattr(child, "tag", "")
        if tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif tag.endswith("}tbl"):
            yield Table(child, doc)


def _table_to_lines(table) -> list[str]:
    lines: list[str] = []
    try:
        rows = table.rows
    except Exception:
        return lines
    for row in rows:
        cells: list[str] = []
        for cell in row.cells:
            text = normalize_whitespace(getattr(cell, "text", "") or "")
            if text:
                cells.append(text)
        if cells:
            lines.append(" | ".join(cells))
    return lines


def parse_docx(path: Path) -> dict | None:
    if not _DOCX_AVAILABLE:
        log.error("DOCX parser not available (install python-docx): %s", path.name)
        return None

    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    sha = sha256_file(path)
    doc_id = make_stable_doc_id(sha256=sha, fallback=str(path))
    title = path.stem.replace("_", " ").replace("-", " ")

    try:
        doc = Document(str(path))
    except Exception as e:
        log.error("Could not open DOCX %s: %s", path.name, e)
        return None

    sections: list[dict] = []
    current_section = "main"
    current_parts: list[str] = []

    for block in _iter_docx_blocks(doc):
        if isinstance(block, Paragraph):
            text = normalize_whitespace(block.text or "")
            if not text:
                continue

            style_name = ""
            try:
                style_name = str(block.style.name or "") if block.style else ""
            except Exception:
                style_name = ""

            if style_name.lower().startswith("heading"):
                if current_parts:
                    body = normalize_whitespace("\n".join(current_parts))
                    if body:
                        sections.append({"section": current_section, "text": body})
                    current_parts = []
                current_section = text[:120] or current_section
            else:
                current_parts.append(text)

        elif isinstance(block, Table):
            lines = _table_to_lines(block)
            if lines:
                current_parts.append("\n".join(lines))

    if current_parts:
        body = normalize_whitespace("\n".join(current_parts))
        if body:
            sections.append({"section": current_section, "text": body})

    if not sections:
        log.warning("No text extracted from DOCX: %s", path.name)
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
        "parser": "python-docx",
        "metadata": {"file_type": "docx", "num_sections": len(sections), "sha256": sha},
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
    sha = sha256_file(path)
    doc_id = make_stable_doc_id(sha256=sha, fallback=str(path))

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
    canonical_url = url_tag["href"] if url_tag and url_tag.get("href") else ""
    url = canonical_url

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
        "metadata": {
            "file_type": "html",
            "num_sections": len(sections),
            "canonical_url": canonical_url,
            "sha256": sha,
        },
    }


def parse_text(path: Path) -> dict | None:
    sha = sha256_file(path)
    doc_id = make_stable_doc_id(sha256=sha, fallback=str(path))
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
        "metadata": {"file_type": path.suffix.lstrip("."), "sha256": sha},
    }


def parse_image(path: Path) -> dict | None:
    """Parse an image via a required sidecar text/markdown file.

    The image itself is not processed. Instead, a sidecar '.txt' or '.md' file
    with the same base name is treated as the source text and flows through the
    normal parsing and indexing pipeline.
    """

    sidecar_txt = path.with_suffix(".txt")
    sidecar_md = path.with_suffix(".md")
    sidecar = sidecar_txt if sidecar_txt.exists() else sidecar_md if sidecar_md.exists() else None

    if sidecar is None:
        log.warning("Image has no sidecar .txt/.md, skipping: %s", path.name)
        return None

    doc = parse_text(sidecar)
    if doc is None:
        return None

    # Make the document title match the image name and attach provenance.
    doc["title"] = path.stem
    meta = doc.get("metadata", {}) or {}
    meta.update({
        "file_type": "image",
        "image_path": str(path),
        "image_ext": path.suffix.lower(),
        "sidecar_path": str(sidecar),
    })
    doc["metadata"] = meta
    return doc


def parse_file(path: Path) -> dict | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return parse_pdf(path)
        elif suffix == ".docx":
            return parse_docx(path)
        elif suffix in (".html", ".htm"):
            return parse_html(path)
        elif suffix in (".txt", ".md"):
            return parse_text(path)
        elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
            return parse_image(path)
        else:
            log.warning("Unsupported file type: %s", path.name)
            return None
    except Exception as e:
        log.error("Unexpected error parsing %s: %s", path.name, e)
        return None


def parse_directory(raw_dir: Path) -> list[dict]:
    extensions = {".pdf", ".docx", ".html", ".htm", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"}
    files = [f for f in raw_dir.rglob("*") if f.suffix.lower() in extensions]
    log.info("Found %d files to parse in %s", len(files), raw_dir)

    manifest_by_rel: dict[str, dict] = {}
    if raw_dir == config.RAW_DIR and config.RAW_MANIFEST_JSONL.exists():
        try:
            for line in config.RAW_MANIFEST_JSONL.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rel = str(rec.get("relative_path", "") or "").strip()
                if rel:
                    manifest_by_rel[rel] = rec
        except Exception as e:
            log.warning("Could not read manifest %s: %s", config.RAW_MANIFEST_JSONL, e)

    documents = []
    for path in files:
        if raw_dir == config.RAW_DIR and path.name == "failed_downloads.txt":
            continue
        doc = parse_file(path)
        if doc:
            if manifest_by_rel:
                try:
                    rel = str(path.relative_to(raw_dir)).replace("\\", "/")
                except Exception:
                    rel = ""

                if rel and rel in manifest_by_rel:
                    rec = manifest_by_rel[rel]
                    src_url = str(rec.get("source_url", "") or "")
                    downloaded_from = str(rec.get("downloaded_from", "") or "")
                    saved_at = str(rec.get("saved_at", "") or "")
                    sha = str(rec.get("sha256", "") or "")

                    if src_url:
                        doc["url"] = src_url
                        doc["doc_id"] = make_stable_doc_id(url=src_url, sha256=sha, fallback=doc.get("source_path", ""))

                    meta = doc.get("metadata", {}) or {}
                    meta.update({
                        "saved_at": saved_at,
                        "downloaded_from": downloaded_from,
                    })
                    if sha:
                        meta["sha256"] = sha
                    doc["metadata"] = meta

            documents.append(doc)
            log.info("Parsed [%s] %s", doc["language"], path.name)
        else:
            log.warning("Skipped: %s", path.name)

    log.info("Successfully parsed %d / %d files", len(documents), len(files))
    return documents
