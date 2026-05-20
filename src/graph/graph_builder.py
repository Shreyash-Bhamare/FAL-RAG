# src/graph/graph_builder.py

import networkx as nx
from typing import List, Dict, Set, Tuple
from src.graph.entity_extractor import LegalEntityExtractor


class KnowledgeGraphBuilder:
    """
    Build a NetworkX knowledge graph from legal chunks.
    
    Graph Structure:
    - Nodes: Legal entities (Articles, Sections, Cases, Concepts)
    - Edges: Relationships between entities
    - Node attributes: article_num, entity_type, description
    - Edge attributes: relationship_type, strength
    """

    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph
        self.extractor = LegalEntityExtractor()

    # ═══════════════════════════════════════════════════════════════
    # BUILD GRAPH
    # ═══════════════════════════════════════════════════════════════

    def build(self, chunks: List[Dict]) -> nx.DiGraph:
        """
        Build the knowledge graph from chunks.
        
        Steps:
        1. Extract entities from all chunks
        2. Add nodes to graph with metadata
        3. Extract relationships
        4. Add edges to graph
        """
        print("\n🔗 Building Knowledge Graph...")

        # Step 1: Extract entities and relationships
        print("   Step 1: Extracting entities and relationships...")
        entities, relationships = self.extractor.process_chunks(chunks)
        print(f"   ✅ Found {len(entities)} entities and {len(relationships)} relationships")

        # Step 2: Add nodes
        print("   Step 2: Adding nodes to graph...")
        self._add_nodes(entities)
        print(f"   ✅ Added {self.graph.number_of_nodes()} nodes")

        # Step 3: Add edges
        print("   Step 3: Adding edges to graph...")
        self._add_edges(relationships)
        print(f"   ✅ Added {self.graph.number_of_edges()} edges")

        # Step 4: Link chunks to entities
        print("   Step 4: Linking chunks to entities...")
        self._link_chunks_to_entities(chunks)
        print(f"   ✅ Linked chunks to graph")

        return self.graph

    # ═══════════════════════════════════════════════════════════════
    # ADD NODES
    # ═══════════════════════════════════════════════════════════════

    def _add_nodes(self, entities: Set[str]) -> None:
        """
        Add nodes to graph with metadata.
        """
        for entity in entities:
            entity_type = self._classify_entity(entity)

            self.graph.add_node(
                entity,
                entity_type=entity_type,
                label=entity.replace("_", " "),
                chunk_ids=[]  # Will be populated later
            )

    def _classify_entity(self, entity: str) -> str:
        """
        Classify entity type based on naming convention.
        """
        if entity.startswith("Article_"):
            return "article"
        elif entity.startswith("Section_"):
            return "section"
        elif entity.startswith("Part_"):
            return "part"
        elif entity.startswith("Case_") or "_v_" in entity:
            return "case"
        else:
            return "concept"

    # ═══════════════════════════════════════════════════════════════
    # ADD EDGES
    # ═══════════════════════════════════════════════════════════════

    def _add_edges(self, relationships: List[Tuple[str, str, str]]) -> None:
        """
        Add edges to graph with relationship metadata.
        """
        for source, rel_type, target in relationships:
            # Add edge if both nodes exist
            if source in self.graph and target in self.graph:
                # Check if edge already exists
                if self.graph.has_edge(source, target):
                    # Increment weight
                    self.graph[source][target]["weight"] += 1
                else:
                    # Create new edge
                    self.graph.add_edge(
                        source,
                        target,
                        relationship_type=rel_type,
                        weight=1
                    )

    # ═══════════════════════════════════════════════════════════════
    # LINK CHUNKS TO ENTITIES
    # ═══════════════════════════════════════════════════════════════

    def _link_chunks_to_entities(self, chunks: List[Dict]) -> None:
        """
        For each chunk, find which entities it contains and link them.
        """
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "unknown")
            article_num = chunk.get("article")

            # Link article entity
            if article_num:
                article_entity = f"Article_{article_num}"
                if article_entity in self.graph:
                    self.graph.nodes[article_entity]["chunk_ids"].append(chunk_id)

            # Extract and link other entities in this chunk
            text = chunk["content"]
            articles = self.extractor.extract_article_references(text)
            sections = self.extractor.extract_section_references(text)
            concepts = self.extractor.extract_legal_concepts(text)

            for entity in articles | sections | concepts:
                if entity in self.graph:
                    if chunk_id not in self.graph.nodes[entity]["chunk_ids"]:
                        self.graph.nodes[entity]["chunk_ids"].append(chunk_id)

    # ═══════════════════════════════════════════════════════════════
    # GRAPH TRAVERSAL
    # ═══════════════════════════════════════════════════════════════

    def get_neighbors(self, entity: str, depth: int = 1) -> List[str]:
        """
        Get all entities connected to a given entity up to specified depth.
        """
        if entity not in self.graph:
            return []

        neighbors = set()

        # BFS traversal up to depth
        visited = set()
        queue = [(entity, 0)]

        while queue:
            current, dist = queue.pop(0)

            if current in visited or dist > depth:
                continue

            visited.add(current)

            # Get successors (entities this one points to)
            for successor in self.graph.successors(current):
                neighbors.add(successor)
                if dist < depth:
                    queue.append((successor, dist + 1))

            # Get predecessors (entities that point to this one)
            for predecessor in self.graph.predecessors(current):
                neighbors.add(predecessor)
                if dist < depth:
                    queue.append((predecessor, dist + 1))

        return list(neighbors)

    def get_entity_chunks(self, entity: str) -> List[str]:
        """
        Get all chunks associated with an entity.
        """
        if entity not in self.graph:
            return []

        return self.graph.nodes[entity].get("chunk_ids", [])

    # ═══════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """
        Get graph statistics.
        """
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "entity_types": {},
            "relationship_types": set(),
            "avg_degree": 0
        }

        # Count entity types
        for node, data in self.graph.nodes(data=True):
            entity_type = data.get("entity_type", "unknown")
            stats["entity_types"][entity_type] = stats["entity_types"].get(entity_type, 0) + 1

        # Count relationship types
        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get("relationship_type", "unknown")
            stats["relationship_types"].add(rel_type)

        # Average degree
        if self.graph.number_of_nodes() > 0:
            stats["avg_degree"] = (
                2 * self.graph.number_of_edges() / self.graph.number_of_nodes()
            )

        return stats

    def print_stats(self) -> None:
        """
        Print graph statistics.
        """
        stats = self.get_stats()

        print(f"\n📊 Knowledge Graph Statistics:")
        print(f"   Nodes              : {stats['num_nodes']}")
        print(f"   Edges              : {stats['num_edges']}")
        print(f"   Avg Degree         : {stats['avg_degree']:.2f}")
        print(f"   Entity Types       : {stats['entity_types']}")
        print(f"   Relationship Types : {sorted(stats['relationship_types'])}")
