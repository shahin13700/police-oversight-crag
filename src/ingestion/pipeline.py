"""
src/ingestion/pipeline.py
-------------------------
Master ingestion function coordinating document parsing.
"""

import os
import logging
from pathlib import Path

from src.ingestion.loader import load_docx, load_pdf
from src.ingestion.chunker import chunk_document, LegislativeChunk
from src.ingestion.reg_chunker import chunk_reg
from src.ingestion.pdf_chunker import chunk_pdf

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def run_ingestion() -> list[LegislativeChunk]:
    """
    Master ingestion function that:
    - Loads and chunks CSPA_2019.docx using existing loader + chunker
    - Loads and chunks all .docx files in data/raw/Regulations/ using reg_chunker
    - Loads and chunks all .pdf files in data/raw/pdfs/ using pdf_chunker
    - Returns a single combined list of LegislativeChunk objects
    """
    all_chunks: list[LegislativeChunk] = []
    
    raw_data_dir = os.getenv("RAW_DATA_DIR", "data/raw")
    base_path = Path(raw_data_dir)
    
    # 1. CSPA_2019.docx
    logger.info("--- Processing CSPA_2019.docx ---")
    try:
        cspa_doc = load_docx("CSPA_2019.docx")
        cspa_chunks = chunk_document(cspa_doc, source_doc="CSPA")
        all_chunks.extend(cspa_chunks)
        logger.info(f"-> CSPA_2019 total chunks: {len(cspa_chunks)}")
    except Exception as e:
        logger.error(f"Error processing CSPA_2019.docx: {e}")

    # 2. Regulations (.docx)
    logger.info("\n--- Processing O. Regulations ---")
    regs_dir = base_path / "Regulations"
    regs_chunks_count = 0
    if regs_dir.exists():
        for file in regs_dir.glob("*.docx"):
            try:
                # load_docx resolves paths relative to RAW_DATA_DIR
                doc = load_docx(f"Regulations/{file.name}")
                chunks = chunk_reg(doc)
                all_chunks.extend(chunks)
                regs_chunks_count += len(chunks)
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")
        logger.info(f"-> O. Regulations total chunks: {regs_chunks_count}")
    else:
        logger.info(f"Directory not found: {regs_dir}")

    # 3. LECA Guidelines (.pdf)
    logger.info("\n--- Processing LECA PDF Guidelines ---")
    pdfs_dir = base_path / "pdfs"
    pdfs_chunks_count = 0
    if pdfs_dir.exists():
        for file in pdfs_dir.glob("*.pdf"):
            try:
                # loader.py load_pdf expects filename, reads from data/raw/pdfs/
                pages = load_pdf(file.name)
                chunks = chunk_pdf(file.name, pages)
                all_chunks.extend(chunks)
                pdfs_chunks_count += len(chunks)
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")
        logger.info(f"-> LECA Guidelines total chunks: {pdfs_chunks_count}")
    else:
        logger.info(f"Directory not found: {pdfs_dir}")

    logger.info("\n==================================================")
    logger.info(f"INGESTION COMPLETE. Total combined chunks: {len(all_chunks)}")
    logger.info("==================================================")

    return all_chunks

if __name__ == "__main__":
    run_ingestion()
