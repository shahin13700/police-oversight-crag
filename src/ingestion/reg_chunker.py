"""
src/ingestion/reg_chunker.py
----------------------------
Splits O. Regulation .docx documents into chunks using LegislativeChunk.
"""

import re
from docx.document import Document as DocumentType
from src.ingestion.chunker import LegislativeChunk

def extract_reg_number(doc: DocumentType) -> str:
    """
    Extracts the regulation number from the document title or early paragraphs.
    e.g. 'ONTARIO REGULATION 392/23' -> '392/23'
    """
    for para in doc.paragraphs[:20]:
        text = para.text.strip()
        match = re.search(r'ONTARIO REGULATION\s+([0-9]+/[0-9]+)', text, re.IGNORECASE)
        if match:
            return match.group(1)
    return "UNKNOWN_REG"

def check_bold_heading(para) -> bool:
    """
    Check if a paragraph acts as a bold heading.
    We consider a paragraph a heading if it has text and meets one of the following criteria:
    1. The paragraph's style name contains "Heading".
    2. The paragraph text is all uppercase and longer than 3 characters.
    """
    text = para.text.strip()
    if not text:
        return False
    
    # 1. Style name contains known heading-related keywords
    if para.style and para.style.name:
        style_lower = para.style.name.lower()
        if "heading" in style_lower or "headnote" in style_lower or "regtitle" in style_lower:
            return True
        
    # 2. All uppercase and > 3 characters
    if len(text) > 3 and text.isupper():
        return True
        
    return False

def chunk_reg(doc: DocumentType) -> list[LegislativeChunk]:
    chunks: list[LegislativeChunk] = []

    reg_number = extract_reg_number(doc)
    source_doc = f"O. Reg. {reg_number}"

    current_part: str = "PREAMBLE"
    current_title: str = ""
    current_section_number: str = ""
    current_lines: list[str] = []
    
    # Regs have part headers e.g. "PART I" / "PART II"
    
    def save_chunk():
        if current_title and current_lines:
            full_text = "\n".join(current_lines).strip()
            if full_text and len(full_text.split()) > 3:
                chunks.append(LegislativeChunk(
                    text=full_text,
                    section_number=current_section_number,
                    section_title=current_title,
                    part_name=current_part,
                    source_doc=source_doc
                ))

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
            
        # Check for PART header
        if re.match(r'^PART\s+[IVXLCDM]+', text):
            save_chunk()
            current_part = text
            # We don't reset title/section yet, wait for the actual heading
            continue
            
        # Check for bold heading
        if check_bold_heading(para):
            save_chunk()
            current_title = text
            current_section_number = ""
            current_lines = [text]
            continue
            
        # Check for section number in body text, e.g., "1.", "2. (1)(a)"
        # Note: Must only extract from the *first* numbered paragraph under a heading
        if current_title and not current_section_number:
            match = re.match(r'^(\d+(?:\.\d+)?)\s*\.\s*(?:\((.*?)\))?', text)
            if not match:
                # Some regulations just start with "1 " or "(1)"
                match = re.match(r'^(\d+)\s+(?:\((.*?)\))?', text)
            
            if match:
                sec_base = match.group(1)
                sec_sub = f"({match.group(2)})" if match.group(2) else ""
                current_section_number = f"{sec_base}{sec_sub}"
                
        if current_title:
            current_lines.append(text)

    save_chunk()
    return chunks
