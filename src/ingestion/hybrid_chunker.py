# src/ingestion/hybrid_chunker.py

import re
from typing import List, Dict, Optional, Tuple


class HybridChunker:
    """
    Improved hybrid chunking strategy based on actual PDF structure.
    
    PDF Structure Observed:
    - Part headers: "PART III FUNDAMENTAL RIGHTS"
    - Article markers: "12 Definition", "13 Laws inconsistent..."
    - Article titles: First line after article number
    - Subsections: (1), (2), (3), (4) with body text
    - Footers: Page numbers, footnotes to be removed
    
    Strategy:
    1. Remove footers (page numbers, footnotes)
    2. Extract articles with improved regex
    3. Keep subsections together (smart subdivision)
    4. Fallback to recursive chunking if needed
    """

    def __init__(self, config: dict):
        self.config = config
        self.max_chunk_size = config["chunking"]["max_chunk_size"]
        self.min_chunk_size = 50
        self.subdivision_threshold = 512

    # ═══════════════════════════════════════════════════════════════
    # PRE-PROCESSING: REMOVE FOOTERS
    # ═══════════════════════════════════════════════════════════════

    def remove_footers(self, text: str) -> str:
        """
        Remove page numbers, footnotes, and other footer artifacts.
        """
        # Remove page numbers (usually single digits at end of lines)
        text = re.sub(r'\n\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
        
        # Remove footnote references like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove "***" or other page break markers
        text = re.sub(r'\*{3,}', '', text)
        
        return text

    # ═══════════════════════════════════════════════════════════════
    # STRATEGY 1: IMPROVED REGEX (STRUCTURE-AWARE)
    # ═══════════════════════════════════════════════════════════════

    def extract_by_improved_regex(self, text: str) -> Optional[List[Dict]]:
        """
        Improved regex based on actual PDF structure.
        
        Captures:
        - Article number (12, 13, 14, etc.)
        - Article title (first line only)
        - Full article body with subsections
        """
        chunks = []

        # More robust pattern that handles actual PDF structure
        # Article number can be at start of line, possibly indented
        # Title is on same line or next line
        # Body extends until next article or part
        article_pattern = re.compile(
            r"(?:^|\n)\s*"                                    # Start of line
            r"(\d+[A-Z]?)"                                    # Article number
            r"\s*[.—\-]?\s*"                                  # Optional separator
            r"([^\n]+?)"                                      # Title (first line only)
            r"\n"                                             # Newline after title
            r"([\s\S]*?)"                                     # Article body
            r"(?=\n\s*(?:\d+[A-Z]?)\s*[.—\-]|\nPART\s+[IVX]+|\Z)",  # Stop at next article/part
            re.MULTILINE | re.IGNORECASE
        )

        matches = list(article_pattern.finditer(text))

        if not matches:
            return None

        # Process each match
        for match in matches:
            article_num = match.group(1).strip()
            raw_title = match.group(2).strip()
            article_body = match.group(3).strip()

            # Clean title: remove trailing punctuation and extra content
            title = self._clean_title(raw_title)

            # Build full content
            full_content = f"Article {article_num}. {title}\n{article_body}"
            token_count = len(full_content.split())

            chunk = {
                "content": full_content,
                "article": article_num,
                "title": title,
                "token_count": token_count,
                "method": "improved_regex"
            }

            chunks.append(chunk)

        # Filter out tiny chunks
        chunks = [c for c in chunks if c["token_count"] >= self.min_chunk_size]

        return chunks if chunks else None

    # ═══════════════════════════════════════════════════════════════
    # SMART SUBDIVISION: SUBSECTION-AWARE
    # ═══════════════════════════════════════════════════════════════

    def smart_subdivide(self, chunk: Dict) -> List[Dict]:
        """
        Intelligently subdivide large articles by subsections.
        
        Only subdivides if:
        - Article >= subdivision_threshold (512 tokens)
        - Subsections exist (numbered (1), (2), (3))
        - Each subsection >= min_chunk_size
        """
        content = chunk["content"]
        token_count = len(content.split())

        # Only subdivide if large enough
        if token_count < self.subdivision_threshold:
            return [chunk]

        # Find subsections: (1), (2), (3), etc.
        subsection_pattern = re.compile(r"\n\s*\((\d+)\)\s*", re.MULTILINE)
        subsections = list(subsection_pattern.finditer(content))

        # If no subsections found, return as-is
        if len(subsections) < 2:
            return [chunk]

        # Extract subsection texts
        sub_chunks = []
        for idx, sub_match in enumerate(subsections):
            sub_num = sub_match.group(1)
            start = sub_match.start()
            end = subsections[idx + 1].start() if idx + 1 < len(subsections) else len(content)

            sub_text = content[start:end].strip()
            sub_token_count = len(sub_text.split())

            # Only include subsections that meet minimum size
            if sub_token_count >= self.min_chunk_size:
                sub_chunk = chunk.copy()
                sub_chunk["content"] = sub_text
                sub_chunk["section"] = sub_num
                sub_chunk["token_count"] = sub_token_count
                sub_chunk["hierarchy_level"] = "subsection"
                sub_chunks.append(sub_chunk)

        # Return subdivided chunks if we got meaningful subdivisions
        return sub_chunks if len(sub_chunks) > 1 else [chunk]

    # ═══════════════════════════════════════════════════════════════
    # TITLE CLEANER
    # ═══════════════════════════════════════════════════════════════

    def _clean_title(self, raw_title: str) -> str:
        """
        Clean article title.
        
        Removes:
        - Trailing punctuation (. — : ,)
        - Repeated words
        - Extra whitespace
        """
        title = raw_title.strip()

        # Remove trailing delimiters
        for delimiter in ['.', '—', ':', '-', ',']:
            if title.endswith(delimiter):
                title = title[:-1].strip()

        # Remove repeated whitespace
        title = re.sub(r'\s+', ' ', title)

        # Limit length
        if len(title) > 120:
            title = title[:120].rsplit(' ', 1)[0]

        return title if len(title) > 3 else "Unknown"

    # ═══════════════════════════════════════════════════════════════
    # STRATEGY 2: RECURSIVE CHUNKING (FALLBACK)
    # ═══════════════════════════════════════════════════════════════

    def recursive_chunk(self, text: str) -> List[Dict]:
        """
        Fallback: Recursive chunking at multiple levels.
        
        Splits at:
        1. Paragraph boundaries (\n\n)
        2. Line boundaries (\n)
        3. Sentence boundaries (. )
        4. Word boundaries ( )
        """
        separators = ["\n\n", "\n", ". ", " "]
        good_chunks = []

        def _split_text(text: str, separators: List[str]) -> List[str]:
            """Recursively split text using separators."""
            final_chunks = []
            separator = separators[-1]

            for _s in separators:
                if _s in text:
                    separator = _s
                    break

            if separator:
                splits = text.split(separator)
            else:
                splits = list(text)

            good_splits = [s.strip() for s in splits if s.strip()]
            return good_splits

        chunks = _split_text(text, separators)

        for idx, chunk_text in enumerate(chunks):
            token_count = len(chunk_text.split())

            if self.min_chunk_size <= token_count <= self.max_chunk_size * 2:
                good_chunks.append({
                    "content": chunk_text,
                    "article": None,
                    "title": chunk_text[:100],
                    "token_count": token_count,
                    "chunk_id": f"recursive_{idx}",
                    "method": "recursive",
                    "hierarchy_level": "text_block"
                })

        return good_chunks

    # ═══════════════════════════════════════════════════════════════
    # HYBRID ORCHESTRATOR
    # ═══════════════════════════════════════════════════════════════

    def chunk(self, text: str) -> Tuple[List[Dict], str]:
        """
        Hybrid chunking with smart subdivision.
        
        Returns:
            (chunks, method_used)
        """
        # Pre-processing: Remove footers
        text = self.remove_footers(text)

        # Strategy 1: Try improved regex
        regex_chunks = self.extract_by_improved_regex(text)

        if regex_chunks and len(regex_chunks) > 5:
            # Regex worked — apply smart subdivision
            print(f"   ✅ Using IMPROVED REGEX method — {len(regex_chunks)} articles found")
            
            final_chunks = []
            for chunk in regex_chunks:
                subdivided = self.smart_subdivide(chunk)
                final_chunks.extend(subdivided)
            
            print(f"   ✂️  Smart subdivision applied — {len(final_chunks)} total chunks")
            return final_chunks, "improved_regex"

        # Strategy 2: Fallback to recursive chunking
        print(f"   ⚠️  Regex method failed — falling back to RECURSIVE")
        recursive_chunks = self.recursive_chunk(text)
        print(f"   ✅ Using RECURSIVE method — {len(recursive_chunks)} chunks")
        return recursive_chunks, "recursive"