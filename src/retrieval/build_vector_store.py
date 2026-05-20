# src/retrieval/build_vector_store.py

from typing import List, Dict
from src.ingestion.pdf_parser import LegalDocumentParser
from src.retrieval.embedder import InLegalBERTEmbedder
from src.retrieval.vector_store import VectorStore


def build_vector_store(
    pdf_path: str,
    config_path: str = "config/config.yaml"
) -> VectorStore:
    """
    Full pipeline to build the vector store from a legal PDF:
    1. Parse PDF and create chunks (Step 1)
    2. Load InLegalBERT embedder (single instance)
    3. Embed all chunks in batches
    4. Store chunks + embeddings in ChromaDB (persistent)
    5. Return searchable vector store

    Args:
        pdf_path   : Path to the legal PDF file
        config_path: Path to config.yaml

    Returns:
        VectorStore: Ready-to-query ChromaDB vector store
    """

    print(f"\n{'═' * 60}")
    print(f"  Building Vector Store — Starting")
    print(f"{'═' * 60}")

    # ── Step 1: Parse PDF and create chunks ───────────────────────
    print(f"\n📄 Step 1: Parsing legal document...")
    parser = LegalDocumentParser(config_path=config_path)
    chunks = parser.process(pdf_path=pdf_path)
    print(f"   ✅ {len(chunks)} chunks ready for embedding")

    # ── Step 2: Load embedder (SINGLE INSTANCE) ───────────────────
    # FIX #1: Load embedder once, pass to VectorStore
    print(f"\n🧠 Step 2: Loading InLegalBERT embedder...")
    embedder = InLegalBERTEmbedder(config_path=config_path)
    dim = embedder.get_embedding_dim()
    print(f"   ✅ Embedding dimension: {dim}")

    # ── Step 3: Embed all chunks ──────────────────────────────────
    print(f"\n🔢 Step 3: Embedding {len(chunks)} chunks in batches of 32...")
    texts = [chunk["content"] for chunk in chunks]
    embeddings = embedder.embed_batch(texts)
    print(f"   ✅ {len(embeddings)} embeddings generated")

    # ── Step 4: Initialize ChromaDB and store ─────────────────────
    print(f"\n💾 Step 4: Storing chunks in ChromaDB...")
    # FIX #1: Pass embedder to VectorStore (don't create new one)
    # fresh_start=True to clear old data before adding new chunks
    store = VectorStore(embedder=embedder, config_path=config_path, fresh_start=True)
    store.add_chunks(chunks=chunks, embeddings=embeddings)

    # ── Step 5: Print stats ───────────────────────────────────────
    store.get_stats()

    print(f"\n{'═' * 60}")
    print(f"  ✅ Vector Store ready — {len(chunks)} documents indexed")
    print(f"{'═' * 60}\n")

    return store


def test_retrieval(store: VectorStore) -> None:
    """
    Run sample queries to validate retrieval quality.
    """
    sample_queries = [
        "What is Article 21?",
        "Right to life and personal liberty",
        "Fundamental rights in Indian Constitution",
        "Constitutional amendments process",
    ]

    print(f"\n{'═' * 60}")
    print(f"  🧪 Testing Retrieval Quality")
    print(f"{'═' * 60}")

    for query in sample_queries:
        print(f"\n🔍 Query: \"{query}\"")
        print(f"{'-' * 60}")

        results = store.search(query, top_k=3)

        for i, result in enumerate(results, 1):
            meta = result["metadata"]
            score = result["similarity_score"]
            content_preview = result["content"][:200].replace("\n", " ")

            print(f"\n  Result {i}:")
            print(f"    Article    : {meta.get('article', 'N/A')}")
            print(f"    Title      : {meta.get('title', 'N/A')}")
            print(f"    Part       : {meta.get('part', 'N/A')}")
            print(f"    Level      : {meta.get('hierarchy_level', 'N/A')}")
            print(f"    Similarity : {score}")
            print(f"    Preview    : {content_preview}...")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT — for standalone testing
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Build the vector store
    store = build_vector_store(
        pdf_path="data/raw/constitution_of_india.pdf",
        config_path="config/config.yaml"
    )

    # Test retrieval quality
    test_retrieval(store)