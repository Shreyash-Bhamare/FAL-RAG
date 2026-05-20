# src/graph/entity_extractor.py

import re
from typing import List, Dict, Set, Tuple


class LegalEntityExtractor:
    """
    Extract legal entities and relationships from chunks.
    
    Entities:
    - Articles (Article 21, Article 14, etc.)
    - Sections (Section 302, Section 300, etc.)
    - Parts (Part III, Part V, etc.)
    - Legal concepts (fundamental rights, equality, etc.)
    
    Relationships:
    - refers_to: "Article 21 refers to Article 14"
    - interprets: "Maneka Gandhi v. Union interprets Article 21"
    - amends: "Amendment X amends Article Y"
    - defined_under: "Right to life is defined under Article 21"
    """

    def __init__(self):
        pass

    # ═══════════════════════════════════════════════════════════════
    # ENTITY EXTRACTION
    # ═══════════════════════════════════════════════════════════════

    def extract_article_references(self, text: str) -> Set[str]:
        """
        Extract all Article references from text.
        Catches: "Article 21", "Article 21 of the Constitution", "Art. 21", etc.
        """
        articles = set()

        # Pattern: Article/Art followed by number
        pattern = r"(?:Article|Art\.?)\s+(\d+[A-Z]?)"
        matches = re.findall(pattern, text, re.IGNORECASE)

        for match in matches:
            articles.add(f"Article_{match}")

        return articles

    def extract_section_references(self, text: str) -> Set[str]:
        """
        Extract all Section references from text.
        Catches: "Section 302", "Section 302 IPC", etc.
        """
        sections = set()

        # Pattern: Section followed by number
        pattern = r"(?:Section|Sec\.?)\s+(\d+[A-Z]?)"
        matches = re.findall(pattern, text, re.IGNORECASE)

        for match in matches:
            sections.add(f"Section_{match}")

        return sections

    def extract_part_references(self, text: str) -> Set[str]:
        """
        Extract Part references from text.
        Catches: "Part III", "Part III Fundamental Rights", etc.
        """
        parts = set()

        # Pattern: Part followed by Roman numerals
        pattern = r"Part\s+([IVX]+[A-Z]?)"
        matches = re.findall(pattern, text, re.IGNORECASE)

        for match in matches:
            parts.add(f"Part_{match}")

        return parts

    def extract_case_references(self, text: str) -> Set[str]:
        """
        Extract Supreme Court case references.
        Catches: "Maneka Gandhi v. Union of India", "1978 SCR 597", etc.
        """
        cases = set()

        # Pattern: "X v. Y" or "X vs Y"
        pattern = r"([A-Za-z\s]+?)\s+(?:v\.|vs\.?)\s+([A-Za-z\s]+?)(?:\s+\(|,|\n)"
        matches = re.findall(pattern, text)

        for match in matches:
            if len(match[0]) > 3 and len(match[1]) > 3:  # Filter out noise
                case_name = f"{match[0].strip()}_v_{match[1].strip()}"
                cases.add(case_name)

        return cases

    def extract_legal_concepts(self, text: str) -> Set[str]:
        """
        Extract legal concepts and keywords.
        Examples: "fundamental rights", "equality", "liberty", "justice", etc.
        """
        concepts = set()

        # Predefined legal concepts relevant to Indian Constitution
        legal_keywords = [
            "fundamental rights",
            "directive principles",
            "equality",
            "liberty",
            "life",
            "personal liberty",
            "freedom of speech",
            "freedom of conscience",
            "freedom of religion",
            "right to property",
            "right to education",
            "right to work",
            "justice",
            "due process",
            "equal protection",
            "state",
            "constitution",
            "parliament",
            "amendment",
        ]

        for keyword in legal_keywords:
            if re.search(r"\b" + keyword + r"\b", text, re.IGNORECASE):
                concepts.add(keyword.replace(" ", "_"))

        return concepts

    # ═══════════════════════════════════════════════════════════════
    # RELATIONSHIP EXTRACTION
    # ═══════════════════════════════════════════════════════════════

    def extract_relationships(self, chunk: Dict) -> List[Tuple[str, str, str]]:
        """
        Extract relationships between entities from a chunk.
        
        Returns list of tuples: (source_entity, relationship_type, target_entity)
        """
        relationships = []
        text = chunk["content"]
        article_num = chunk.get("article")

        # Extract entities
        articles = self.extract_article_references(text)
        sections = self.extract_section_references(text)
        cases = self.extract_case_references(text)
        concepts = self.extract_legal_concepts(text)

        # Create relationships based on proximity and patterns

        # 1. Article to Concept relationships
        if article_num:
            source = f"Article_{article_num}"
            for concept in concepts:
                relationships.append((source, "discusses", concept))

        # 2. Article references within chunk
        article_list = list(articles)
        for i, article1 in enumerate(article_list):
            for article2 in article_list[i + 1:]:
                # Check if articles are mentioned close to each other (relationship)
                if self._are_close_in_text(article1, article2, text):
                    relationships.append((article1, "related_to", article2))

        # 3. Case references
        if cases and article_num:
            source = f"Article_{article_num}"
            for case in cases:
                relationships.append((source, "interpreted_by", case))

        # 4. Section references (if in legal text like IPC)
        if sections and article_num:
            source = f"Article_{article_num}"
            for section in sections:
                relationships.append((source, "refers_to", section))

        return relationships

    def _are_close_in_text(self, entity1: str, entity2: str, text: str, window: int = 500) -> bool:
        """
        Check if two entities are close to each other in the text (within window chars).
        """
        entity1_clean = entity1.replace("_", " ")
        entity2_clean = entity2.replace("_", " ")

        pos1 = text.find(entity1_clean)
        pos2 = text.find(entity2_clean)

        if pos1 == -1 or pos2 == -1:
            return False

        distance = abs(pos1 - pos2)
        return distance < window

    # ═══════════════════════════════════════════════════════════════
    # PROCESS CHUNKS
    # ═══════════════════════════════════════════════════════════════

    def process_chunks(self, chunks: List[Dict]) -> Tuple[Set[str], List[Tuple[str, str, str]]]:
        """
        Extract all entities and relationships from all chunks.
        
        Returns:
            (all_entities, all_relationships)
        """
        all_entities = set()
        all_relationships = []

        for chunk in chunks:
            # Extract entities
            articles = self.extract_article_references(chunk["content"])
            sections = self.extract_section_references(chunk["content"])
            parts = self.extract_part_references(chunk["content"])
            cases = self.extract_case_references(chunk["content"])
            concepts = self.extract_legal_concepts(chunk["content"])

            # Add to global entity set
            all_entities.update(articles)
            all_entities.update(sections)
            all_entities.update(parts)
            all_entities.update(cases)
            all_entities.update(concepts)

            # Extract relationships from this chunk
            rels = self.extract_relationships(chunk)
            all_relationships.extend(rels)

        return all_entities, all_relationships
