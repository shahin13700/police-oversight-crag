"""
src/ingestion/chunker.py
------------------------
Splits a python-docx Document into chunks that respect the legislative
structure of the CSPA (Community Safety and Policing Act, 2019).

WHY WE DON'T CHUNK BY TOKEN COUNT:
Legal documents are structured around sections and subsections that carry
specific legal meaning. Splitting mid-section destroys that meaning and makes
accurate citation (e.g. "CSPA s.11(1)") impossible. Each chunk must map to
exactly one legislative section so we can cite it precisely.

CHUNKING STRATEGY (based on actual CSPA .docx paragraph styles):
- A new chunk starts at every 'headnote' style paragraph (section title)
- The chunk collects all paragraphs below it until the next headnote
- Part names (style: 'partnum') are tracked and attached as metadata
- Section numbers are extracted from the first 'section' style paragraph

STYLES FOUND IN CSPA .docx:
- 'partnum'    → Part header e.g. "PART I"
- 'headnote'   → Section title e.g. "Declaration of principles"
- 'section'    → Section body e.g. "1 Policing shall be provided..."
- 'subsection' → Subsection body e.g. "(2) For the purposes of..."
- 'definition' → Definition entry
- 'paragraph'  → Sub-items (a), (b), (c)...

"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from docx.document import Document as DocumentType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model for a single legislative chunk
# ---------------------------------------------------------------------------

@dataclass
class LegislativeChunk:
    """
    Represents one section or subsection of legislation as a self-contained chunk.

    Each chunk carries enough metadata to:
    1. Generate an accurate citation (e.g. "CSPA s.11(1)")
    2. Filter by part or source document
    3. Display the section title in the UI alongside the answer

    Fields:
        text:             Full text of the section including heading.
        section_number:   e.g. "11", "11(1)", "11(1)(a)" — extracted from section body.
        section_title:    e.g. "Adequate and effective policing" — from headnote style.
        part_name:        e.g. "PART III — PROVISION OF POLICING" — from partnum style.
        source_doc:       Identifier for the source document e.g. "CSPA_2019".
    """
    text: str
    section_number: str
    section_title: str
    part_name: str
    source_doc: str

    def citation(self) -> str:
        """
        Returns a formatted legal citation string.
        """
        base_doc = "CSPA" if "CSPA" in self.source_doc else self.source_doc
        if self.section_number:
            return f"{base_doc} s.{self.section_number}"
        return f"{base_doc} — {self.section_title}"

    def __repr__(self) -> str:
        return (
            f"LegislativeChunk("
            f"citation='{self.citation()}', "
            f"title='{self.section_title[:40]}', "
            f"words={len(self.text.split())})"
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_section_number(text: str) -> str:
    """
    Extract the section number from the beginning of a section paragraph.

    Legislative sections in the CSPA follow these patterns:
      "1 Policing shall..."          → "1"
      "2 (1)  In this Act,"          → "2(1)"
      "11 (1)  Adequate and..."      → "11(1)"
      "(2)  For the purposes of..."  → subsection only, returns ""

    We extract the leading number pattern and normalise it (remove spaces
    between section and subsection numbers).

    Args:
        text: Raw paragraph text from a 'section' style paragraph.

    Returns:
        Normalised section number string, or empty string if not found.
    """
    # Pattern: optional leading number, optional subsection in brackets
    # Matches: "1", "2 (1)", "11 (1)", "(2)" etc.
    match = re.match(r'^(\d+)\s*(\(\d+\))?', text.strip())
    if match:
        section = match.group(1)
        subsection = match.group(2) or ""
        # Normalise: "11 (1)" → "11(1)"
        return f"{section}{subsection}"
    return ""


def clean_part_name(text: str) -> str:
    """
    Clean and normalise a part name from a 'partnum' style paragraph.

    The CSPA uses multi-line part headers like:
      "PART III\nRESPONSIBILITY FOR PROVIDING POLICING"

    We join them into a single clean string.

    Args:
        text: Raw text from a partnum paragraph (may contain newlines).

    Returns:
        Cleaned part name string.
    """
    # Replace newlines with " — " to produce "PART III — RESPONSIBILITY FOR..."
    return re.sub(r'\s*\n\s*', ' — ', text.strip())


# ---------------------------------------------------------------------------
# Main chunker function
# ---------------------------------------------------------------------------

def chunk_document(
    doc: DocumentType,
    source_doc: str = "CSPA_2019"
) -> list[LegislativeChunk]:
    """
    Split a python-docx Document into LegislativeChunk objects.

    The algorithm works in a single pass through all paragraphs:
    1. Track the current Part name whenever we see a 'partnum' paragraph
    2. When we hit a 'headnote' paragraph, save the current chunk (if any)
       and start a new one with that headnote as the section title
    3. Accumulate all subsequent paragraphs into the current chunk's text
    4. Extract the section number from the first 'section' paragraph we see
    5. If no section number found, inherit from previous section + detect subsection

    Args:
        doc:        A python-docx Document object (from loader.load_docx()).
        source_doc: Identifier string for the source document.

    Returns:
        List of LegislativeChunk objects, one per legislative section.
    """
    chunks: list[LegislativeChunk] = []

    current_part: str = "PREAMBLE"
    current_title: str = ""
    current_section_number: str = ""
    last_known_section_number: str = ""  # full e.g. "79(1)"
    last_known_section_base: str = ""    # base only e.g. "79"
    current_lines: list[str] = []
    found_section_num: bool = False

    def resolve_section_number() -> str:
        """
        Resolve the best section number for the current chunk.

        If the chunk has its own section number, use it.
        If not, try to extract a subsection number from the first line
        of text and combine with the last known base section number.
        e.g. base="79", first line="(3) A chief of police..." → "79(3)"
        """
        if current_section_number:
            return current_section_number

        if not last_known_section_base:
            return last_known_section_number

        # Look for subsection pattern at start of accumulated text
        if current_lines:
            # Check second line (first line is the title)
            text_to_check = current_lines[1] if len(current_lines) > 1 else current_lines[0]
            subsection_match = re.match(r'^\s*\((\d+(?:\.\d+)?)\)', text_to_check.strip())
            if subsection_match:
                return f"{last_known_section_base}({subsection_match.group(1)})"

        return last_known_section_number

    def save_current_chunk() -> None:
        """Save the accumulated lines as a LegislativeChunk."""
        if current_title and current_lines:
            full_text = "\n".join(current_lines).strip()
            if full_text and len(full_text.split()) > 5:
                effective_section = resolve_section_number()
                chunks.append(LegislativeChunk(
                    text=full_text,
                    section_number=effective_section,
                    section_title=current_title,
                    part_name=current_part,
                    source_doc=source_doc,
                ))

    for para in doc.paragraphs:
        style = para.style.name
        text = para.text.strip()

        if not text:
            continue

        # ── PART HEADER ────────────────────────────────────────────────────
        if style == "partnum":
            save_current_chunk()
            current_part = clean_part_name(text)
            current_title = ""
            current_section_number = ""
            current_lines = []
            found_section_num = False
            continue

        # ── SECTION TITLE (starts a new chunk) ─────────────────────────────
        if style == "headnote":
            save_current_chunk()
            current_title = text
            current_section_number = ""
            current_lines = [text]
            found_section_num = False
            continue

        # ── SECTION BODY (extract section number from the first one) ────────
        if style == "section" and not found_section_num:
            current_section_number = extract_section_number(text)
            found_section_num = True
            if current_section_number:
                last_known_section_number = current_section_number
                # Extract base e.g. "79(1)" → "79"
                base_match = re.match(r'^(\d+)', current_section_number)
                if base_match:
                    last_known_section_base = base_match.group(1)

        # ── ALL OTHER CONTENT ───────────────────────────────────────────────
        skip_styles = {
            "shorttitle", "chapter", "ConsolidationPeriod",
            "comment", "footnoteLeft", "toc", "Normal",
        }

        if style not in skip_styles and current_title:
            current_lines.append(text)

    save_current_chunk()

    if chunks:
        word_counts = [len(c.text.split()) for c in chunks]
        logger.info(
            f"[chunker] Produced {len(chunks)} chunks from '{source_doc}'\n"
            f"          Smallest: {min(word_counts)} words | "
            f"Largest: {max(word_counts)} words | "
            f"Average: {sum(word_counts) // len(word_counts)} words"
        )
    else:
        logger.warning(f"[chunker] WARNING: No chunks produced from '{source_doc}'")

    return chunks