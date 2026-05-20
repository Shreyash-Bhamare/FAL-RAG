# src/ingestion/pdf_parser.py

import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from src.utils.helpers import load_config, clean_text
from src.ingestion.hybrid_chunker import HybridChunker


class LegalDocumentParser:
    """
    Legal document parser with hybrid chunking strategy.
    
    Chunking approach:
    1. Try hierarchy-based regex (legal structure)
    2. Fallback to recursive chunking if regex fails
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.max_chunk_size = self.config["chunking"]["max_chunk_size"]
        
        # Initialize hybrid chunker
        self.chunker = HybridChunker(config=self.config)

        # Populated dynamically during processing
        self.document_title = "Unknown Legal Document"
        self.toc_map = {}
        self.part_positions = []

    # ═══════════════════════════════════════════════════════════════
    # PHASE A: PDF PARSING
    # ═══════════════════════════════════════════════════════════════

    def parse_pdf(self, pdf_path: str) -> str:
        """Extract raw text from PDF using PyMuPDF."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        raw_pages = []
        page_count = len(doc)

        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                raw_pages.append(f"[PAGE_{page_num + 1}]\n{text}")

        
        full_text = "\n".join(raw_pages)
        print(f"   ✅ Parsed {page_count} pages from: {pdf_path.name}")
        doc.close()
        return full_text

    def clean_raw_text(self, text: str) -> str:
        """Normalize whitespace and fix PDF artifacts."""
        text = re.sub(r"\[PAGE_\d+\]", "", text)
        text = re.sub(r"-\n", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        return text.strip()

    # ═══════════════════════════════════════════════════════════════
    # DOCUMENT TITLE EXTRACTION
    # ═══════════════════════════════════════════════════════════════

    def extract_document_title(self, text: str) -> str:
        """Extract document title from PDF header."""
        header_text = text[:1000]

        title_patterns = [
            r"THE\s+([A-Z][A-Z\s]+(?:ACT|CODE|CONSTITUTION|ORDINANCE|RULES|REGULATIONS)(?:\s*,\s*\d{4})?)",
            r"([A-Z][A-Z\s]+(?:ACT|CODE|CONSTITUTION|ORDINANCE)(?:\s*,\s*\d{4})?)",
            r"CONSTITUTION\s+OF\s+INDIA",
            r"([A-Z]{3,}(?:\s+[A-Z]{2,}){1,6})",
        ]

        for pattern in title_patterns:
            match = re.search(pattern, header_text)
            if match:
                title = match.group(0).strip()
                title = re.sub(r"\s+", " ", title)
                if len(title) > 5:
                    print(f"   ✅ Document title detected: {title}")
                    return title

        print(f"   ⚠️  Could not detect document title — using default")
        return "Unknown Legal Document"

    # ═══════════════════════════════════════════════════════════════
    # PHASE B: HYBRID CHUNKING (NEW)
    # ═══════════════════════════════════════════════════════════════

    def chunk_text(self, text: str) -> Tuple[List[Dict], str]:
        """
        Use hybrid chunking: regex first, recursive fallback.
        Returns: (chunks, method_used)
        """
        chunks, method = self.chunker.chunk(text)

        # Add metadata to chunks
        for chunk in chunks:
            if "chunk_id" not in chunk:
                # Generate chunk_id if not already set
                article = chunk.get("article", "unknown")
                chunk["chunk_id"] = f"article_{article}_{len(chunks)}"
            
            chunk["document_title"] = self.document_title
            chunk["part"] = "Unknown"  # Will be filled if needed
            chunk["section"] = chunk.get("section", None)
            chunk["hierarchy_level"] = chunk.get("hierarchy_level", "unknown")

        return chunks, method

    # ═══════════════════════════════════════════════════════════════
    # PHASE D: VALIDATION & QUALITY CHECKS
    # ═══════════════════════════════════════════════════════════════

    def validate_chunks(self, chunks: List[Dict]) -> Dict:
        """Generate quality report on chunks."""
        total_chunks = len(chunks)
        if total_chunks == 0:
            print("⚠️  No chunks created. Check PDF parsing.")
            return {}

        token_counts = [c["token_count"] for c in chunks]
        total_tokens = sum(token_counts)
        avg_tokens = total_tokens / total_chunks
        over_limit = sum(1 for t in token_counts if t > self.max_chunk_size)

        report = {
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": round(avg_tokens, 1),
            "max_chunk_tokens": max(token_counts),
            "min_chunk_tokens": min(token_counts),
            "chunks_exceeding_limit": over_limit,
        }

        print(f"\n✅ Chunking Quality Report:")
        print(f"   Document Title        : {self.document_title}")
        print(f"   Total Chunks          : {report['total_chunks']}")
        print(f"   Total Tokens          : {report['total_tokens']}")
        print(f"   Avg Tokens / Chunk    : {report['avg_tokens_per_chunk']}")
        print(f"   Max Chunk Tokens      : {report['max_chunk_tokens']}")
        print(f"   Min Chunk Tokens      : {report['min_chunk_tokens']}")
        print(f"   Chunks Over Limit     : {report['chunks_exceeding_limit']}")

        return report

    def sample_chunks(self, chunks: List[Dict], num_samples: int = 5) -> None:
        """Print sample chunks for manual review."""
        import random
        samples = random.sample(chunks, min(num_samples, len(chunks)))

        print(f"\n📋 Sample Chunks for Manual Review:")
        for i, chunk in enumerate(samples, 1):
            print(f"\n{'═' * 60}")
            print(f"  Sample         : {i}")
            print(f"  Chunk ID       : {chunk.get('chunk_id', 'N/A')}")
            print(f"  Document       : {chunk.get('document_title', 'N/A')}")
            print(f"  Article        : {chunk.get('article', 'N/A')}")
            print(f"  Tokens         : {chunk['token_count']}")
            print(f"  Method         : {chunk.get('method', 'N/A')}")
            print(f"{'-' * 60}")
            preview = chunk["content"][:500]
            print(preview + ("..." if len(chunk["content"]) > 500 else ""))

    # ═══════════════════════════════════════════════════════════════
    # MAIN PROCESS PIPELINE
    # ═══════════════════════════════════════════════════════════════

    def process(self, pdf_path: str) -> List[Dict]:
        """
        Full pipeline:
        Parse → Clean → Extract Title → Hybrid Chunk → Validate → Return
        """
        print(f"\n{'═' * 60}")
        print(f"  Legal Document Parser — Starting (Hybrid Chunking)")
        print(f"{'═' * 60}")

        # Phase A: Parse & Clean
        print("\n📄 Phase A: Parsing PDF...")
        raw_text = self.parse_pdf(pdf_path)

        print("\n🧹 Cleaning raw text...")
        clean = self.clean_raw_text(raw_text)

        # Extract document title
        print("\n📌 Extracting document title...")
        self.document_title = self.extract_document_title(clean)

        # Phase B: Hybrid chunking (NEW)
        print("\n✂️  Phase B: Hybrid chunking (regex + recursive fallback)...")
        chunks, method = self.chunk_text(clean)
        print(f"   Chunking method used: {method.upper()}")

        # Phase D: Validate
        print("\n🧪 Phase C: Validating chunks...")
        report = self.validate_chunks(chunks)

        self.sample_chunks(chunks, num_samples=5)

        print(f"\n{'═' * 60}")
        print(f"  ✅ Parsing complete — {len(chunks)} chunks ready")
        print(f"{'═' * 60}\n")

        return chunks


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT — for standalone testing
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = LegalDocumentParser(config_path="config/config.yaml")
    chunks = parser.process(pdf_path="data/raw/constitution_of_india.pdf")
    print(f"\nTotal chunks returned: {len(chunks)}")