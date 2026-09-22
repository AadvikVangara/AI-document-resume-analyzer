"""
document_parser.py
-------------------
Utility module for extracting text and computing document statistics
from various file formats (PDF, DOCX, TXT).
"""

import io
from typing import Dict, Any, Tuple
from pypdf import PdfReader
from docx import Document


def extract_text_from_file(file_bytes: bytes, file_name: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extracts text and metadata from an uploaded file bytes object.
    
    Supported formats:
    - .pdf  (via pypdf)
    - .docx (via python-docx)
    - .txt  (via standard utf-8 decoding with latin-1 fallback)
    
    Returns:
        tuple: (extracted_text, metadata_dict)
    """
    lower_name = file_name.lower()
    text = ""
    page_count = 1

    try:
        if lower_name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file_bytes))
            page_count = len(reader.pages)
            extracted_pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted_pages.append(f"--- [Page {i + 1}] ---\n{page_text}")
            text = "\n\n".join(extracted_pages)

        elif lower_name.endswith(".docx"):
            doc = Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)
            # Rough estimation for docx: ~350 words per page
            word_est = len(text.split())
            page_count = max(1, (word_est // 350) + 1)

        elif lower_name.endswith(".txt"):
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1", errors="replace")
            word_est = len(text.split())
            page_count = max(1, (word_est // 400) + 1)

        else:
            raise ValueError(f"Unsupported file format: {file_name}. Please upload PDF, DOCX, or TXT.")

    except Exception as e:
        raise RuntimeError(f"Error parsing file '{file_name}': {str(e)}")

    text = text.strip()
    stats = compute_text_stats(text, page_count)
    return text, stats


def compute_text_stats(text: str, page_count: int = 1) -> Dict[str, Any]:
    """
    Computes statistical metrics for the extracted document text.
    Helpful for understanding token limits and reading time.
    """
    if not text:
        return {
            "characters": 0,
            "words": 0,
            "estimated_tokens": 0,
            "reading_time_mins": 0,
            "pages": 0,
        }

    char_count = len(text)
    words = text.split()
    word_count = len(words)
    # Average rule of thumb: 1 token ~= 0.75 words, or 1 word ~= 1.33 tokens
    estimated_tokens = int(word_count * 1.33)
    # Average adult reading speed: ~200 words per minute
    reading_time = max(1, round(word_count / 200))

    return {
        "characters": char_count,
        "words": word_count,
        "estimated_tokens": estimated_tokens,
        "reading_time_mins": reading_time,
        "pages": page_count,
    }
