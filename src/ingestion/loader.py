"""
src/ingestion/loader.py
-----------------------
Responsible for loading a .docx legislation file from disk and returning
a python-docx Document object for the chunker to process.

This module does ONE thing only: open the file safely.
All parsing and chunking logic lives in chunker.py.

"""

import os
import logging
from pathlib import Path

from docx import Document

from docx.document import Document as DocumentType
from dotenv import load_dotenv

import pdfplumber

logger = logging.getLogger(__name__)

# Load environment variables from the .env file at the repo root.
# This allows us to read RAW_DATA_DIR without hardcoding any paths.
load_dotenv()


def load_docx(filename: str) -> DocumentType:
    """
    Load a .docx file from the raw data directory and return a Document object.

    The raw data directory is read from the RAW_DATA_DIR environment variable
    defined in .env. This means no file paths are ever hardcoded in the code,
    making the project portable across different machines and team members.

    Args:
        filename (str): The name of the .docx file to load, e.g. "CSPA_2019.docx".
                        Do not include the full path — just the filename.

    Returns:
        DocumentType: A python-docx Document object ready for the chunker to parse.

    Raises:
        FileNotFoundError: If the file does not exist at the expected path.
                           The error message includes the full path so the user
                           knows exactly where to place the file.
        ValueError: If the filename does not end in .docx, since python-docx
                    cannot process .doc (old Word 97-2003 format) files.

    Example:
        >>> doc = load_docx("CSPA_2019.docx")
        >>> print(f"Loaded {len(doc.paragraphs)} paragraphs")
    """

    # Validate that the caller passed a .docx file, not a .doc file.
    # python-docx will crash with a confusing error on .doc files, so we
    # catch this early and give a clear, helpful message instead.
    if not filename.endswith(".docx"):
        raise ValueError(
            f"File '{filename}' is not a .docx file. "
            "python-docx only supports .docx format. "
            "If you have a .doc file, open it in Microsoft Word and "
            "save it as Word Document (.docx) first."
        )

    # Read the raw data directory from environment variables.
    # Falls back to "data/raw" if RAW_DATA_DIR is not set in .env,
    # which matches our repo structure and is safe for all team members.
    raw_data_dir = os.getenv("RAW_DATA_DIR", "data/raw")

    # Build the full path to the file using pathlib.
    # pathlib.Path handles Windows/Mac/Linux path differences automatically,
    # so this works on every team member's machine regardless of OS.
    file_path = Path(raw_data_dir) / filename

    # Check the file exists before trying to open it.
    # Without this check, python-docx raises a cryptic PackageNotFoundError
    # that doesn't tell the user what went wrong. This gives a clear message.
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find '{filename}' at expected path: {file_path.resolve()}\n"
            f"Please download the CSPA .docx from ontario.ca/laws and place it in "
            f"the '{raw_data_dir}/' folder. Do NOT commit the file to git."
        )

    # Open and return the Document object.
    # We don't parse anything here — that's the chunker's job.
    # Keeping this function focused on a single responsibility makes it
    # easier to test and easier to swap out if we change document formats later.
    doc = Document(str(file_path))

    # Log a confirmation so the user knows the file loaded successfully.
    # This is helpful during development and debugging.
    logger.info(f"[loader] Successfully loaded '{filename}' ({len(doc.paragraphs)} paragraphs)")

    return doc


def load_pdf(filename: str) -> list[str]:
    """
    Load a .pdf file from the raw data directory and extract text per page.

    Args:
        filename (str): The name of the .pdf file to load.

    Returns:
        list[str]: A list of strings, where each string represents the text of a single page.
    """
    if not filename.endswith(".pdf"):
        raise ValueError(
            f"File '{filename}' is not a .pdf file."
        )

    raw_data_dir = os.getenv("RAW_DATA_DIR", "data/raw")
    file_path = Path(raw_data_dir) / "pdfs" / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find '{filename}' at expected path: {file_path.resolve()}"
        )

    pages_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
            else:
                pages_text.append("")

    logger.info(f"[loader] Successfully loaded '{filename}' ({len(pages_text)} pages text extracted)")
    return pages_text
