"""Multi-format document loader with advanced extraction capabilities.

Supported formats and strategies:

| Category    | Extensions                | Strategy / Library                         |
|-------------|---------------------------|--------------------------------------------|
| PDF         | .pdf                      | pdfplumber (tables+text), fallback PyPDF   |
| Word        | .docx                     | docx2txt / python-docx                     |
| Excel       | .xlsx, .xls, .csv         | openpyxl / pandas → text tables            |
| PowerPoint  | .pptx                     | python-pptx slide text + notes             |
| Markdown    | .md                       | UnstructuredMarkdownLoader                 |
| Plain text  | .txt, .log, .json, .xml   | TextLoader (UTF-8)                         |
| HTML / Web  | .html, .htm               | BeautifulSoup text extraction              |
| Images/OCR  | .png, .jpg, .jpeg, .tiff, .bmp | pytesseract OCR                       |
| Audio/Video | .mp3, .wav, .mp4, .m4a, .webm  | OpenAI Whisper transcription          |

Each loader returns ``List[Document]`` with rich metadata (source, page,
file_type, extraction_method).  All external dependencies are lazily
imported so the system degrades gracefully if optional packages are
missing.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from app.utils.exceptions import DocumentLoadError, UnsupportedFileTypeError
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------

def _meta(file_path: str, **extra: Any) -> Dict[str, Any]:
    """Build base metadata dict."""
    p = Path(file_path)
    return {
        "source": p.name,
        "file_type": p.suffix.lower().lstrip("."),
        **extra,
    }


# ---------------------------------------------------------------------------
# PDF loader — pdfplumber with table extraction, fallback to PyPDF
# ---------------------------------------------------------------------------

def _load_pdf(file_path: str) -> List[Document]:
    docs: List[Document] = []
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                parts: List[str] = []

                # Extract text
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(text)

                # Extract tables → Markdown tables
                tables = page.extract_tables() or []
                for table in tables:
                    if not table:
                        continue
                    md_table = _table_to_markdown(table)
                    if md_table:
                        parts.append(f"\n[表格]\n{md_table}")

                if parts:
                    content = "\n".join(parts)
                    docs.append(Document(
                        page_content=content,
                        metadata=_meta(file_path, page=page_num, extraction_method="pdfplumber"),
                    ))

        if docs:
            logger.info("PDF loaded via pdfplumber: %d pages from %s", len(docs), file_path)
            return docs
    except ImportError:
        logger.info("pdfplumber not available, falling back to PyPDF")
    except Exception as exc:
        logger.warning("pdfplumber failed for %s: %s. Falling back to PyPDF.", file_path, exc)

    # Fallback: PyPDF
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update(_meta(file_path, extraction_method="pypdf"))
        logger.info("PDF loaded via PyPDF: %d pages from %s", len(docs), file_path)
        return docs
    except Exception as exc:
        raise DocumentLoadError(f"PDF load failed: {exc}") from exc


def _table_to_markdown(table: List[List[Optional[str]]]) -> str:
    """Convert a 2D table to Markdown format."""
    if not table or len(table) < 1:
        return ""
    # Clean cells
    cleaned = []
    for row in table:
        cleaned.append([str(cell).strip() if cell else "" for cell in row])

    lines = []
    # Header
    lines.append("| " + " | ".join(cleaned[0]) + " |")
    lines.append("| " + " | ".join(["---"] * len(cleaned[0])) + " |")
    # Body
    for row in cleaned[1:]:
        # Pad row if shorter than header
        while len(row) < len(cleaned[0]):
            row.append("")
        lines.append("| " + " | ".join(row[:len(cleaned[0])]) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Word loader (.docx)
# ---------------------------------------------------------------------------

def _load_docx(file_path: str) -> List[Document]:
    try:
        import docx2txt  # type: ignore
        text = docx2txt.process(file_path)
        if text and text.strip():
            return [Document(
                page_content=text,
                metadata=_meta(file_path, extraction_method="docx2txt"),
            )]
        return []
    except ImportError:
        pass

    # Fallback: python-docx for more structure
    try:
        from docx import Document as DocxDocument  # type: ignore
        doc = DocxDocument(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        if paragraphs:
            return [Document(
                page_content="\n".join(paragraphs),
                metadata=_meta(file_path, extraction_method="python-docx"),
            )]
        return []
    except ImportError:
        raise DocumentLoadError("Neither docx2txt nor python-docx is installed")
    except Exception as exc:
        raise DocumentLoadError(f"DOCX load failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Excel loader (.xlsx, .xls, .csv)
# ---------------------------------------------------------------------------

def _load_excel(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()
    try:
        import openpyxl  # type: ignore
    except ImportError:
        openpyxl = None

    docs: List[Document] = []

    if ext == ".csv":
        return _load_csv(file_path)

    if ext in (".xlsx", ".xls"):
        # Try openpyxl first
        if openpyxl:
            try:
                wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    rows = []
                    for row in ws.iter_rows(values_only=True):
                        rows.append([str(cell) if cell is not None else "" for cell in row])
                    if rows:
                        md = _table_to_markdown(rows)
                        if md:
                            docs.append(Document(
                                page_content=f"## Sheet: {sheet_name}\n\n{md}",
                                metadata=_meta(file_path, sheet=sheet_name, extraction_method="openpyxl"),
                            ))
                wb.close()
                if docs:
                    logger.info("Excel loaded via openpyxl: %d sheet(s) from %s", len(docs), file_path)
                    return docs
            except Exception as exc:
                logger.warning("openpyxl failed: %s", exc)

        # Fallback: pandas
        try:
            import pandas as pd  # type: ignore
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                text = f"## Sheet: {sheet_name}\n\n{df.to_markdown(index=False)}"
                docs.append(Document(
                    page_content=text,
                    metadata=_meta(file_path, sheet=sheet_name, extraction_method="pandas"),
                ))
            if docs:
                return docs
        except ImportError:
            raise DocumentLoadError("openpyxl or pandas required for Excel files")
        except Exception as exc:
            raise DocumentLoadError(f"Excel load failed: {exc}") from exc

    return docs


def _load_csv(file_path: str) -> List[Document]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        if rows:
            md = _table_to_markdown(rows)
            return [Document(
                page_content=md,
                metadata=_meta(file_path, extraction_method="csv_reader"),
            )]
        return []
    except Exception as exc:
        raise DocumentLoadError(f"CSV load failed: {exc}") from exc


# ---------------------------------------------------------------------------
# PowerPoint loader (.pptx)
# ---------------------------------------------------------------------------

def _load_pptx(file_path: str) -> List[Document]:
    try:
        from pptx import Presentation  # type: ignore
    except ImportError:
        raise DocumentLoadError("python-pptx is required for .pptx files")

    try:
        prs = Presentation(file_path)
        docs: List[Document] = []
        for slide_num, slide in enumerate(prs.slides, 1):
            parts: List[str] = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text = shape.text_frame.text.strip()
                    if text:
                        parts.append(text)
                if shape.has_table:
                    table = shape.table
                    rows = []
                    for row in table.rows:
                        rows.append([cell.text.strip() for cell in row.cells])
                    md = _table_to_markdown(rows)
                    if md:
                        parts.append(md)
            # Speaker notes
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    parts.append(f"[演讲者备注] {notes}")

            if parts:
                docs.append(Document(
                    page_content="\n".join(parts),
                    metadata=_meta(file_path, slide=slide_num, extraction_method="python-pptx"),
                ))
        logger.info("PPTX loaded: %d slide(s) from %s", len(docs), file_path)
        return docs
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"PPTX load failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Markdown loader
# ---------------------------------------------------------------------------

def _load_markdown(file_path: str) -> List[Document]:
    try:
        from langchain_community.document_loaders import UnstructuredMarkdownLoader
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update(_meta(file_path, extraction_method="unstructured_md"))
        return docs
    except ImportError:
        # Fallback: load as plain text
        return _load_text(file_path)
    except Exception as exc:
        raise DocumentLoadError(f"Markdown load failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Plain text / structured text loader
# ---------------------------------------------------------------------------

def _load_text(file_path: str) -> List[Document]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if not text.strip():
            return []
        return [Document(
            page_content=text,
            metadata=_meta(file_path, extraction_method="text_reader"),
        )]
    except Exception as exc:
        raise DocumentLoadError(f"Text load failed: {exc}") from exc


def _load_json(file_path: str) -> List[Document]:
    """Load JSON and convert to readable text."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = json.dumps(data, ensure_ascii=False, indent=2)
        return [Document(
            page_content=text,
            metadata=_meta(file_path, extraction_method="json_reader"),
        )]
    except Exception as exc:
        raise DocumentLoadError(f"JSON load failed: {exc}") from exc


# ---------------------------------------------------------------------------
# HTML / web page loader
# ---------------------------------------------------------------------------

def _load_html(file_path: str) -> List[Document]:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        # Fallback: strip tags with regex
        return _load_html_fallback(file_path)

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Extract tables separately
        parts: List[str] = []
        for table_tag in soup.find_all("table"):
            rows = []
            for tr in table_tag.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                rows.append(cells)
            if rows:
                parts.append(_table_to_markdown(rows))
            table_tag.decompose()

        # Main text
        main_text = soup.get_text(separator="\n", strip=True)
        if main_text:
            parts.insert(0, main_text)

        content = "\n\n".join(parts)
        if content.strip():
            return [Document(
                page_content=content,
                metadata=_meta(file_path, extraction_method="beautifulsoup"),
            )]
        return []
    except Exception as exc:
        raise DocumentLoadError(f"HTML load failed: {exc}") from exc


def _load_html_fallback(file_path: str) -> List[Document]:
    import re
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        return [Document(page_content=text, metadata=_meta(file_path, extraction_method="regex_strip"))]
    return []


# ---------------------------------------------------------------------------
# OCR loader (scanned images / PDFs)
# ---------------------------------------------------------------------------

def _load_ocr_image(file_path: str) -> List[Document]:
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except ImportError:
        raise DocumentLoadError(
            "Pillow and pytesseract are required for OCR. "
            "Install: pip install Pillow pytesseract"
        )

    try:
        image = Image.open(file_path)
        # Attempt Chinese + English OCR
        text = pytesseract.image_to_string(image, lang="chi_sim+eng")
        if text and text.strip():
            return [Document(
                page_content=text.strip(),
                metadata=_meta(file_path, extraction_method="pytesseract_ocr"),
            )]
        return []
    except Exception as exc:
        raise DocumentLoadError(f"OCR failed: {exc}") from exc


def _load_scanned_pdf(file_path: str) -> List[Document]:
    """For PDFs that are scanned images — convert pages to images then OCR."""
    try:
        from pdf2image import convert_from_path  # type: ignore
        import pytesseract  # type: ignore
    except ImportError:
        raise DocumentLoadError(
            "pdf2image and pytesseract are required for scanned PDF OCR. "
            "Install: pip install pdf2image pytesseract"
        )

    try:
        images = convert_from_path(file_path, dpi=300)
        docs: List[Document] = []
        for page_num, img in enumerate(images, 1):
            text = pytesseract.image_to_string(img, lang="chi_sim+eng")
            if text and text.strip():
                docs.append(Document(
                    page_content=text.strip(),
                    metadata=_meta(file_path, page=page_num, extraction_method="pdf_ocr"),
                ))
        logger.info("Scanned PDF OCR: %d page(s) from %s", len(docs), file_path)
        return docs
    except Exception as exc:
        raise DocumentLoadError(f"Scanned PDF OCR failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Audio / Video transcription (Whisper)
# ---------------------------------------------------------------------------

def _load_audio_video(file_path: str, whisper_model: str = "base") -> List[Document]:
    """Transcribe audio/video files using OpenAI Whisper."""
    try:
        import whisper  # type: ignore
    except ImportError:
        raise DocumentLoadError(
            "openai-whisper is required for audio/video transcription. "
            "Install: pip install openai-whisper"
        )

    try:
        logger.info("Transcribing with Whisper (model=%s): %s", whisper_model, file_path)
        model = whisper.load_model(whisper_model)
        result = model.transcribe(file_path, language=None)  # auto-detect language

        text = result.get("text", "")
        language = result.get("language", "unknown")
        segments = result.get("segments", [])

        docs: List[Document] = []

        if not text or not text.strip():
            return []

        # If long, split by segments for better chunking
        if len(segments) > 10:
            # Group segments into ~60 second windows
            current_parts: List[str] = []
            current_start = 0.0
            window_size = 60.0  # seconds

            for seg in segments:
                seg_text = seg.get("text", "").strip()
                seg_start = seg.get("start", 0.0)
                if not seg_text:
                    continue

                if seg_start - current_start > window_size and current_parts:
                    docs.append(Document(
                        page_content=" ".join(current_parts),
                        metadata=_meta(
                            file_path,
                            start_time=current_start,
                            end_time=seg_start,
                            language=language,
                            extraction_method="whisper",
                        ),
                    ))
                    current_parts = []
                    current_start = seg_start

                current_parts.append(seg_text)

            # Remaining
            if current_parts:
                docs.append(Document(
                    page_content=" ".join(current_parts),
                    metadata=_meta(
                        file_path,
                        start_time=current_start,
                        language=language,
                        extraction_method="whisper",
                    ),
                ))
        else:
            docs.append(Document(
                page_content=text.strip(),
                metadata=_meta(file_path, language=language, extraction_method="whisper"),
            ))

        logger.info(
            "Whisper transcription: %d segment(s), lang=%s from %s",
            len(docs), language, file_path,
        )
        return docs
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"Audio/video transcription failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Unstructured universal fallback
# ---------------------------------------------------------------------------

def _load_unstructured(file_path: str) -> List[Document]:
    """Last-resort loader using the `unstructured` library's auto-detection."""
    try:
        from unstructured.partition.auto import partition  # type: ignore
    except ImportError:
        raise DocumentLoadError(
            "unstructured library is required as fallback. "
            "Install: pip install unstructured"
        )

    try:
        elements = partition(filename=file_path)
        text = "\n\n".join(str(el) for el in elements if str(el).strip())
        if text.strip():
            return [Document(
                page_content=text,
                metadata=_meta(file_path, extraction_method="unstructured_auto"),
            )]
        return []
    except Exception as exc:
        raise DocumentLoadError(f"Unstructured fallback failed: {exc}") from exc


# ===================================================================
# Main factory
# ===================================================================

# Extension → loader function mapping
_LOADER_REGISTRY: Dict[str, Any] = {
    # PDF
    ".pdf": _load_pdf,
    # Word
    ".docx": _load_docx,
    # Excel / CSV
    ".xlsx": _load_excel,
    ".xls": _load_excel,
    ".csv": _load_excel,
    # PowerPoint
    ".pptx": _load_pptx,
    # Markdown
    ".md": _load_markdown,
    # Plain text / structured
    ".txt": _load_text,
    ".log": _load_text,
    ".json": _load_json,
    ".xml": _load_text,
    ".yaml": _load_text,
    ".yml": _load_text,
    # HTML
    ".html": _load_html,
    ".htm": _load_html,
    # Images (OCR)
    ".png": _load_ocr_image,
    ".jpg": _load_ocr_image,
    ".jpeg": _load_ocr_image,
    ".tiff": _load_ocr_image,
    ".tif": _load_ocr_image,
    ".bmp": _load_ocr_image,
    # Audio / Video (Whisper)
    ".mp3": _load_audio_video,
    ".wav": _load_audio_video,
    ".m4a": _load_audio_video,
    ".mp4": _load_audio_video,
    ".webm": _load_audio_video,
    ".flac": _load_audio_video,
    ".ogg": _load_audio_video,
}


class DocumentLoaderFactory:
    """Create the appropriate document loader based on file extension.

    Supports 25+ file extensions across 8 categories:
    PDF, Word, Excel/CSV, PowerPoint, Markdown, HTML, Images (OCR),
    Audio/Video (Whisper transcription).
    """

    @staticmethod
    def supported_extensions() -> List[str]:
        return sorted(_LOADER_REGISTRY.keys())

    @staticmethod
    def supported_categories() -> Dict[str, List[str]]:
        return {
            "PDF (含表格提取)": [".pdf"],
            "Word 文档": [".docx"],
            "Excel / CSV": [".xlsx", ".xls", ".csv"],
            "PowerPoint": [".pptx"],
            "Markdown": [".md"],
            "纯文本 / 结构化": [".txt", ".log", ".json", ".xml", ".yaml", ".yml"],
            "HTML 网页": [".html", ".htm"],
            "图片 (OCR)": [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"],
            "音视频 (Whisper 转写)": [".mp3", ".wav", ".m4a", ".mp4", ".webm", ".flac", ".ogg"],
        }

    @staticmethod
    def load(file_path: str, enable_ocr_fallback: bool = True) -> List[Document]:
        """Load a document from the given file path.

        Args:
            file_path: Path to the file.
            enable_ocr_fallback: If True, attempt OCR on PDFs that yield
                no text from normal extraction.

        Returns:
            List of Document objects with text content and metadata.
        """
        ext = Path(file_path).suffix.lower()
        loader_fn = _LOADER_REGISTRY.get(ext)

        if loader_fn is None:
            # Try unstructured as universal fallback
            try:
                logger.info("Unknown extension %s, trying unstructured fallback", ext)
                docs = _load_unstructured(file_path)
                if docs:
                    return docs
            except Exception:
                pass
            raise UnsupportedFileTypeError(ext)

        try:
            docs = loader_fn(file_path)

            # PDF OCR fallback: if PDF yielded no text, try scanned-PDF OCR
            if ext == ".pdf" and enable_ocr_fallback and not docs:
                logger.info("PDF yielded no text, attempting OCR fallback for %s", file_path)
                try:
                    docs = _load_scanned_pdf(file_path)
                except Exception as ocr_exc:
                    logger.warning("OCR fallback also failed: %s", ocr_exc)

            if not docs:
                logger.warning("No content extracted from %s", file_path)
                docs = [Document(
                    page_content="[文件内容为空或无法提取]",
                    metadata=_meta(file_path, extraction_method="empty"),
                )]

            logger.info(
                "Loaded %d document(s) from %s [%s]",
                len(docs),
                file_path,
                docs[0].metadata.get("extraction_method", "?") if docs else "?",
            )
            return docs

        except (UnsupportedFileTypeError, DocumentLoadError):
            raise
        except Exception as exc:
            logger.error("Failed to load %s: %s", file_path, exc)
            raise DocumentLoadError(f"Failed to load {file_path}: {exc}") from exc
