# src/retrieval/vector_store.py

import chromadb
from typing import List, Dict, Optional
from src.utils.helpers import load_config
import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

class VectorStore:
    """
    ChromaDB wrapper for storing and retrieving legal document chunks.
    
    Fixes:
    - No double model loading (embedder passed in)
    - Persistent storage enabled
    - Unique ID generation to avoid duplicates
    """

    def __init__(self, embedder, config_path: str = "config/config.yaml", fresh_start: bool = False):
        import os
        os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
        
        self.config = load_config(config_path)
        self.persist_dir = self.config["paths"]["chroma_db"]
        self.collection_name = "legal_documents"
        self.embedder = embedder

        print(f"   Initializing ChromaDB (persistent)...")
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        if fresh_start:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"   🔄 Cleared existing collection (fresh start)")
            except Exception:
                pass
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"   ✅ ChromaDB initialized")
        print(f"   Collection    : {self.collection_name}")
        print(f"   Storage       : Persistent (path: {self.persist_dir})")

    # ═══════════════════════════════════════════════════════════════
    # ADD CHUNKS TO VECTOR STORE
    # ═══════════════════════════════════════════════════════════════

    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]]
    ) -> None:
        """
        Add chunks and their embeddings to ChromaDB.
        Stores rich metadata for each chunk.
        
        FIX #4: Ensures unique IDs by adding global index.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                f"must be the same length."
            )

        # Prepare data for ChromaDB
        ids = []
        docs = []
        metas = []

        for idx, chunk in enumerate(chunks):
            # FIX #4: Make IDs globally unique by adding index
            unique_id = f"{chunk['chunk_id']}_{idx}"
            
            ids.append(unique_id)
            docs.append(chunk["content"])
            metas.append({
                "chunk_id":        chunk.get("chunk_id", ""),
                "document_title":  chunk.get("document_title", ""),
                "part":            chunk.get("part", ""),
                "article":         str(chunk.get("article", "")),
                "section":         str(chunk.get("section", "")),
                "title":           chunk.get("title", ""),
                "hierarchy_level": chunk.get("hierarchy_level", ""),
                "token_count":     str(chunk.get("token_count", 0)),
            })

        # Add to ChromaDB in one batch
        self.collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metas
        )

        print(f"   ✅ Added {len(chunks)} chunks to ChromaDB")

    # ═══════════════════════════════════════════════════════════════
    # SEARCH VECTOR STORE
    # ═══════════════════════════════════════════════════════════════

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Embed query and search ChromaDB for top-k similar chunks.
        Returns list of results with content, metadata and similarity score.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # Embed the query using the shared embedder
        query_embedding = self.embedder.embed_text(query)

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # Convert cosine distance to similarity score
            similarity = round(1 - dist, 4)
            formatted.append({
                "content":         doc,
                "metadata":        meta,
                "similarity_score": similarity
            })

        return formatted

    # ═══════════════════════════════════════════════════════════════
    # STATS
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """Return statistics about the vector store."""
        count = self.collection.count()
        stats = {
            "collection_name": self.collection_name,
            "total_documents": count,
            "storage": f"Persistent ({self.persist_dir})",
        }

        print(f"\n📊 Vector Store Stats:")
        print(f"   Collection    : {stats['collection_name']}")
        print(f"   Total Docs    : {stats['total_documents']}")
        print(f"   Storage       : {stats['storage']}")

        return stats