"""
src/ingestion/pdf_chunker.py
----------------------------
Splits LECA guideline PDFs into chunks using LegislativeChunk.
"""

import re
from pathlib import Path
from src.ingestion.chunker import LegislativeChunk


def extract_pdf_source_doc(filename: str) -> str:
    """
    Extracts canonical source_doc name from a LECA PDF filename or slug.
    Supports both standardized slugs (LECA_001_...) and legacy filenames.
    """
    filename_clean = Path(filename).stem

    # 1. Standardized slug format: LECA_001_Reviewing_Complaints
    match_slug = re.match(r"^LECA_(\d+)_(.*)", filename_clean, re.IGNORECASE)
    if match_slug:
        num = match_slug.group(1)
        title = match_slug.group(2).replace("_", " ").replace("-", " ").strip()
        return f"LECA Guideline {num} \u2014 {title}"

    # 2. Legacy filename format: 001-Guideline for Reviewing Complaints
    match_legacy = re.match(r"^(\d+)[-\u2013\s]+Guideline\s+(?:for\s+)?(.*)", filename_clean, re.IGNORECASE)
    if match_legacy:
        num = match_legacy.group(1)
        title = match_legacy.group(2).replace("-", " ").strip()
        return f"LECA Guideline {num} \u2014 {title}"

    # 3. Rules of Procedure
    if "rules" in filename_clean.lower():
        return "LECA Rules"

    # 4. Fallback
    return f"LECA \u2014 {filename_clean.replace('_', ' ').strip()}"


def chunk_pdf(filename: str, pages: list[str]) -> list[LegislativeChunk]:
    chunks: list[LegislativeChunk] = []
    source_doc = extract_pdf_source_doc(filename)

    current_title: str = ""
    current_lines: list[str] = []
    current_part: str = "GENERAL"

    def save_chunk():
        if current_title and current_lines:
            full_text = "\n".join(current_lines).strip()
            if full_text and len(full_text.split()) > 3:
                chunks.append(LegislativeChunk(
                    text=full_text,
                    section_number="",  # Guidelines mostly use sections as bullet points under headings
                    section_title=current_title,
                    part_name=current_part,
                    source_doc=source_doc
                ))

    # Concatenate all pages and split by lines
    all_lines = []
    for page in pages:
        all_lines.extend(page.splitlines())

    for line in all_lines:
        line = line.strip()
        if not line:
            continue

        # Heuristic for headings: ALL CAPS or Title Case without trailing punctuation
        is_heading = False
        if line.isupper() and len(line) < 100:
            is_heading = True
        elif not line.endswith(".") and not line.endswith(":") and len(line) > 2 and len(line.split()) < 10:
            # Check if majority of words are capitalized
            words = [w for w in line.split() if len(w) > 3]
            if words and all(w[0].isupper() for w in words):
                is_heading = True

        if is_heading:
            save_chunk()
            current_title = line
            current_lines = [line]
            continue

        if current_title:
            current_lines.append(line)

    save_chunk()
    return chunks
