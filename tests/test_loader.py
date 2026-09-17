"""
tests/test_loader.py
--------------------
Unit tests for src/ingestion/loader.py.

We test three things:
1. The loader successfully loads a real .docx file
2. The loader raises FileNotFoundError for a missing file
3. The loader raises ValueError for a non-.docx file

We use a small sample .docx created in memory for testing so the tests
don't depend on the real CSPA file being present (which is gitignored).

"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch
from docx import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_sample_docx(path: Path) -> None:
    """
    Create a minimal .docx file at the given path for testing purposes.
    This avoids any dependency on the real CSPA file during tests.
    """
    doc = Document()
    doc.add_heading("Community Safety and Policing Act, 2019", level=1)
    doc.add_heading("Part I — Interpretation", level=2)
    doc.add_paragraph("1 (1) In this Act, 'board' means a police services board.")
    doc.add_paragraph("(2) Words in the singular include the plural.")
    doc.save(str(path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadDocx:
    """Tests for the load_docx() function."""

    def test_loads_valid_docx_successfully(self, tmp_path):
        """
        Test that load_docx() returns a Document object when given a valid
        .docx file in the expected directory.
        """
        # Arrange: create a sample .docx in a temp directory
        sample_file = tmp_path / "CSPA_2019.docx"
        create_sample_docx(sample_file)

        # Patch RAW_DATA_DIR to point to our temp directory
        with patch.dict(os.environ, {"RAW_DATA_DIR": str(tmp_path)}):
            # Import here so the patched env var is picked up
            from src.ingestion.loader import load_docx

            # Act
            doc = load_docx("CSPA_2019.docx")

        # Assert: we got a Document back with paragraphs in it
        assert doc is not None
        assert len(doc.paragraphs) > 0

    def test_raises_file_not_found_for_missing_file(self, tmp_path):
        """
        Test that load_docx() raises a clear FileNotFoundError when the
        requested file does not exist in the data directory.
        """
        # Patch RAW_DATA_DIR to an empty temp directory (no files in it)
        with patch.dict(os.environ, {"RAW_DATA_DIR": str(tmp_path)}):
            from src.ingestion.loader import load_docx

            # Act & Assert: expect FileNotFoundError with a helpful message
            with pytest.raises(FileNotFoundError) as exc_info:
                load_docx("CSPA_2019.docx")

        # The error message should tell the user where to put the file
        assert "ontario.ca/laws" in str(exc_info.value)

    def test_raises_value_error_for_non_docx_file(self, tmp_path):
        """
        Test that load_docx() raises ValueError when given a .doc file
        (old Word 97-2003 format) instead of a .docx file.

        This is important because python-docx silently fails or gives
        a confusing error on .doc files.
        """
        with patch.dict(os.environ, {"RAW_DATA_DIR": str(tmp_path)}):
            from src.ingestion.loader import load_docx

            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                load_docx("CSPA_2019.doc")

        # The error message should tell the user how to fix the problem
        assert ".docx" in str(exc_info.value)
        assert "Microsoft Word" in str(exc_info.value)
