"""
src/ingestion/pdf_chunker.py
----------------------------
Splits LECA guideline PDFs into chunks using LegislativeChunk.
"""

import re
from pathlib import Path
from src.ingestion.chunker import LegislativeChunk

def chunk_pdf(filename: str, pages: list[str]) -> list[LegislativeChunk]:
    chunks: list[LegislativeChunk] = []
    
    # Extract guideline name from filename
    filename_clean = Path(filename).stem
    
    # E.g. 001-Guideline for Reviewing Complaints -> LECA Guideline 001 — Reviewing Complaints
    match = re.match(r'^(\d+)[-–\s]+Guideline\s+(?:for\s+)?(.*)', filename_clean, re.IGNORECASE)
    if match:
        doc_name = f"LECA Guideline {match.group(1)} — {match.group(2).replace('-', ' ').strip()}"
    elif "Rules of Procedure" in filename_clean:
        doc_name = "LECA Rules"
    else:
        doc_name = f"LECA — {filename_clean}"
        
    source_doc = doc_name
    
    current_title: str = ""
    current_lines: list[str] = []
    
    # PDFs don't typically have "Parts" in the same way, we can use "GENERAL"
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

    # We concatenate all pages and split by lines
    all_lines = []
    for page in pages:
        all_lines.extend(page.splitlines())
        
    for line in all_lines:
        line = line.strip()
        if not line:
            continue
            
        # Heuristic for headings: ALL CAPS or Title Case without trailing punctuation
        # Also limit length since headings are usually short
        is_heading = False
        if line.isupper() and len(line) < 100:
            is_heading = True
        elif not line.endswith('.') and not line.endswith(':') and len(line) > 2 and len(line.split()) < 10:
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
