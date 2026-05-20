# src/retrieval/dual_retriever.py

from typing import List, Dict
import networkx as nx
from src.retrieval.vector_store import VectorStore
from src.graph.graph_builder import KnowledgeGraphBuilder


class DualRetriever:
    """
    Dual retrieval: Vector similarity + Graph traversal.
    
    Strategy:
    1. Vector search (semantic similarity) → top-k chunks
    2. Graph traversal from retrieved articles → related articles
    3. Merge and rank results by combined score
    """

    def __init__(self, vector_store: VectorStore, graph_builder: KnowledgeGraphBuilder):
        self.vector_store = vector_store
        self.graph = graph_builder.graph
        self.builder = graph_builder

    # ═══════════════════════════════════════════════════════════════
    # VECTOR SEARCH
    # ═══════════════════════════════════════════════════════════════

    def vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve chunks using vector similarity.
        Returns: List of (chunk, similarity_score)
        """
        return self.vector_store.search(query, top_k=top_k)

    # ═══════════════════════════════════════════════════════════════
    # GRAPH SEARCH
    # ═══════════════════════════════════════════════════════════════

    def graph_search(self, article_nums: List[str], depth: int = 1, top_k: int = 5) -> List[Dict]:
        """
        Traverse graph from given articles to find related articles.
        
        Args:
            article_nums: List of article numbers (e.g., ["21", "14"])
            depth: How deep to traverse (1 = direct neighbors)
            top_k: How many results to return
        
        Returns: List of related articles with their chunks
        """
        related_articles = {}

        for article_num in article_nums:
            entity = f"Article_{article_num}"

            if entity not in self.graph:
                continue

            # Get neighbors via graph traversal
            neighbors = self.builder.get_neighbors(entity, depth=depth)

            for neighbor in neighbors:
                if neighbor not in related_articles:
                    # Extract article number from entity
                    if neighbor.startswith("Article_"):
                        related_articles[neighbor] = {
                            "entity": neighbor,
                            "chunks": self.builder.get_entity_chunks(neighbor),
                            "depth": self._get_distance(entity, neighbor)
                        }

        # Sort by depth (closer = better) and take top-k
        sorted_articles = sorted(
            related_articles.items(),
            key=lambda x: x[1]["depth"]
        )[:top_k]

        # Convert to chunk format
        results = []
        for entity, data in sorted_articles:
            for chunk_id in data["chunks"]:
                results.append({
                    "entity": entity,
                    "chunk_id": chunk_id,
                    "graph_distance": data["depth"]
                })

        return results

    def _get_distance(self, source: str, target: str) -> int:
        """
        Get shortest path distance between two entities in graph.
        """
        try:
            return nx.shortest_path_length(self.graph, source, target)
        except nx.NetworkXNoPath:
            return float('inf')

    # ═══════════════════════════════════════════════════════════════
    # DUAL RETRIEVAL
    # ═══════════════════════════════════════════════════════════════

    def retrieve(self, query: str, top_k: int = 5, use_graph: bool = True) -> List[Dict]:
        """
        Dual retrieval: Combine vector + graph search.
        
        Steps:
        1. Vector search → get top-k similar chunks
        2. Extract article numbers from results
        3. Graph traversal → get related articles
        4. Merge results, rank by combined score
        """
        results = {}

        # Step 1: Vector search
        vector_results = self.vector_search(query, top_k=top_k)

        for i, result in enumerate(vector_results):
            meta = result["metadata"]
            chunk_id = meta.get("chunk_id", "unknown")
            article = meta.get("article", "unknown")
            score = result["similarity_score"]

            # Rank by vector similarity (1 - rank/top_k for decreasing weight)
            vector_weight = 1.0 - (i / top_k)

            results[chunk_id] = {
                "metadata": meta,
                "content": result["content"],
                "vector_score": score,
                "vector_weight": vector_weight,
                "graph_weight": 0,
                "final_score": vector_weight,
                "source": "vector"
            }

        # Step 2: Graph search (if enabled)
        if use_graph:
            article_nums = [
                r["metadata"].get("article")
                for r in vector_results
                if r["metadata"].get("article")
            ]

            if article_nums:
                graph_results = self.graph_search(article_nums, depth=1, top_k=top_k)

                for graph_result in graph_results:
                    chunk_id = graph_result.get("chunk_id", "unknown")

                    if chunk_id not in results:
                        # New chunk from graph search
                        graph_distance = graph_result.get("graph_distance", float('inf'))
                        graph_weight = 1.0 / (1.0 + graph_distance)  # Closer = higher weight

                        results[chunk_id] = {
                            "metadata": {"article": graph_result["entity"]},
                            "content": "",
                            "vector_score": 0,
                            "vector_weight": 0,
                            "graph_weight": graph_weight,
                            "final_score": graph_weight * 0.5,  # Graph results weighted lower
                            "source": "graph"
                        }
                    else:
                        # Boost existing chunk with graph signal
                        graph_distance = graph_result.get("graph_distance", float('inf'))
                        graph_weight = 1.0 / (1.0 + graph_distance)
                        results[chunk_id]["graph_weight"] = graph_weight
                        results[chunk_id]["final_score"] = (
                            results[chunk_id]["vector_weight"] * 0.7 +
                            graph_weight * 0.3
                        )

        # Step 3: Rank by combined score
        ranked = sorted(
            results.items(),
            key=lambda x: x[1]["final_score"],
            reverse=True
        )[:top_k]

        # Return as list
        final_results = []
        for chunk_id, data in ranked:
            final_results.append({
                "chunk_id": chunk_id,
                "metadata": data["metadata"],
                "content": data["content"],
                "vector_score": round(data["vector_score"], 4),
                "graph_weight": round(data["graph_weight"], 4),
                "final_score": round(data["final_score"], 4),
                "source": data["source"]
            })

        return final_results

    # ═══════════════════════════════════════════════════════════════
    # SEARCH WITH RETRIEVAL DETAILS
    # ═══════════════════════════════════════════════════════════════

    def search(self, query: str, top_k: int = 5, verbose: bool = False) -> List[Dict]:
        """
        Convenience method for dual retrieval with optional verbosity.
        """
        results = self.retrieve(query, top_k=top_k, use_graph=True)

        if verbose:
            print(f"\n🔍 Query: \"{query}\"")
            print(f"   Results: {len(results)}")
            for i, r in enumerate(results, 1):
                print(f"   {i}. Article {r['metadata'].get('article', 'N/A')} "
                      f"(vector: {r['vector_score']}, graph: {r['graph_weight']}, "
                      f"final: {r['final_score']}) [{r['source']}]")

        return results
