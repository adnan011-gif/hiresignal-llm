"""
File text extraction utilities for HireSignal.
Supports PDF, images (OCR), DOCX, and plain text files.
"""

import os

import fitz  # pymupdf
import pytesseract
from PIL import Image
from docx import Document


def extract_from_pdf(file_path: str) -> str:
    """Extract text from all pages of a PDF using pymupdf (fitz)."""
    doc = fitz.open(file_path)
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()
    return "\n".join(pages_text).strip()


def extract_from_image(file_path: str) -> str:
    """Extract text from an image file using Tesseract OCR."""
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return text.strip()


def extract_from_docx(file_path: str) -> str:
    """Extract all paragraph text from a Word document."""
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs).strip()


def extract_text_from_file(file_path: str) -> str:
    """
    Master extraction function — detects file type by extension
    and routes to the correct extractor.

    Supported extensions: .pdf, .png, .jpg, .jpeg, .docx, .txt

    Raises:
        ValueError: If the file type is not supported.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_from_pdf(file_path)
    elif ext in (".png", ".jpg", ".jpeg"):
        return extract_from_image(file_path)
    elif ext == ".docx":
        return extract_from_docx(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported types: .pdf, .png, .jpg, .jpeg, .docx, .txt"
        )
