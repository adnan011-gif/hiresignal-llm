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
    """Extract text from all pages of a PDF using pymupdf (fitz).
    Falls back to OCR via pytesseract if no selectable text is found."""
    try:
        doc = fitz.open(file_path)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        text = "\n".join(pages_text).strip()

        if not text:
            # OCR fallback: render each page as an image and OCR it
            print(f"[file_extractor] No selectable text in PDF, attempting OCR fallback for: {file_path}")
            ocr_pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img).strip()
                if page_text:
                    ocr_pages.append(page_text)
            text = "\n".join(ocr_pages).strip()

        doc.close()

        if not text:
            raise ValueError(
                "No text could be extracted from this PDF — neither selectable text nor OCR produced results. "
                "The PDF may contain only non-text graphics."
            )
        print(f"[file_extractor] PDF extracted {len(text)} chars from: {file_path}")
        return text
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error extracting text from PDF: {e}")


def extract_from_image(file_path: str) -> str:
    """Extract text from an image file using Tesseract OCR."""
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        text = text.strip()
        if not text:
            raise ValueError("No text detected in the image.")
        return text
    except Exception as e:
        raise RuntimeError(f"Error performing OCR on image: {e}")


def extract_from_docx(file_path: str) -> str:
    """Extract all paragraph text from a Word document."""
    try:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs]
        text = "\n".join(paragraphs).strip()
        if not text:
            raise ValueError("No text could be extracted from this Word document.")
        return text
    except Exception as e:
        raise RuntimeError(f"Error extracting text from DOCX: {e}")


def extract_text_from_file(file_path: str) -> str:
    """
    Master extraction function — detects file type by extension
    and routes to the correct extractor.

    Supported extensions: .pdf, .png, .jpg, .jpeg, .docx, .txt

    Raises:
        ValueError: If the file does not exist, exceeds the size limit, or has an unsupported format.
        RuntimeError: If the file cannot be read or parsed.
    """
    if not file_path:
        raise ValueError("No file path provided.")

    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    try:
        file_size = os.path.getsize(file_path)
    except Exception as e:
        raise RuntimeError(f"Could not read file size: {e}")

    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    if file_size > MAX_SIZE:
        raise ValueError(f"File size ({file_size / (1024*1024):.2f}MB) exceeds the 10MB limit.")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_from_pdf(file_path)
    elif ext in (".png", ".jpg", ".jpeg"):
        return extract_from_image(file_path)
    elif ext == ".docx":
        return extract_from_docx(file_path)
    elif ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                raise ValueError("Plain text file is empty.")
            return text
        except Exception as e:
            raise RuntimeError(f"Could not read text file: {e}")
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported types: .pdf, .png, .jpg, .jpeg, .docx, .txt"
        )
