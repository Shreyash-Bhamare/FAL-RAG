# src/graph/build_graph.py

from src.ingestion.pdf_parser import LegalDocumentParser
from src.graph.graph_builder import KnowledgeGraphBuilder


def build_knowledge_graph(
    pdf_path: str,
    config_path: str = "config/config.yaml"
):
    """
    Full pipeline to build knowledge graph from legal PDF.
    
    Steps:
    1. Parse PDF and create chunks
    2. Build knowledge graph (extract entities + relationships)
    3. Return graph
    """

    print(f"\n{'═' * 60}")
    print(f"  Building Knowledge Graph — Starting")
    print(f"{'═' * 60}")

    # Step 1: Parse PDF
    print(f"\n📄 Step 1: Parsing legal document...")
    parser = LegalDocumentParser(config_path=config_path)
    chunks = parser.process(pdf_path=pdf_path)
    print(f"   ✅ {len(chunks)} chunks created")

    # Step 2: Build graph
    print(f"\n🔗 Step 2: Building knowledge graph...")
    graph_builder = KnowledgeGraphBuilder()
    graph = graph_builder.build(chunks)

    # Step 3: Print statistics
    graph_builder.print_stats()

    print(f"\n{'═' * 60}")
    print(f"  ✅ Knowledge graph built successfully")
    print(f"{'═' * 60}\n")

    return graph, graph_builder


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    graph, builder = build_knowledge_graph(
        pdf_path="data/raw/constitution_of_india.pdf",
        config_path="config/config.yaml"
    )

    # Test graph traversal
    print("\n🧪 Testing graph traversal:")
    
    test_entities = ["Article_21", "Article_14", "Part_III"]
    
    for entity in test_entities:
        neighbors = builder.get_neighbors(entity, depth=1)
        chunks = builder.get_entity_chunks(entity)
        
        print(f"\n  {entity}:")
        print(f"    Neighbors (depth 1) : {len(neighbors)}")
        print(f"    Chunks             : {len(chunks)}")
        
        if neighbors:
            print(f"    Connected to       : {neighbors[:5]}")
