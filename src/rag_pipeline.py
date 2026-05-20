# src/rag_pipeline.py

from src.ingestion.pdf_parser import LegalDocumentParser
from src.retrieval.embedder import InLegalBERTEmbedder
from src.retrieval.vector_store import VectorStore
from src.graph.graph_builder import KnowledgeGraphBuilder
from src.retrieval.dual_retriever import DualRetriever
from src.generation.llm_generator import LLMGenerator
from src.hallucination.span_detector import SpanHallucinationDetector
from src.hallucination.faithfulness_scorer import FaithfulnessScorer


class FALRAgPipeline:

    def __init__(self, config_path: str = "config/config.yaml"):
        print("Initializing embedder...")
        self.embedder = InLegalBERTEmbedder(config_path=config_path)

        print("Initializing vector store...")
        self.vector_store = VectorStore(embedder=self.embedder, config_path=config_path, fresh_start=False)

        print("Initializing graph builder...")
        self.graph_builder = KnowledgeGraphBuilder()

        print("Initializing dual retriever...")
        self.retriever = DualRetriever(self.vector_store, self.graph_builder)

        print("Initializing LLM generator...")
        self.generator = LLMGenerator(config_path=config_path)

        print("Initializing hallucination detector...")
        self.detector = SpanHallucinationDetector(self.embedder)

        print("Initializing faithfulness scorer...")
        self.scorer = FaithfulnessScorer()

    def process_query(self, query: str, top_k: int = 5) -> dict:
        retrieval_results = self.retriever.retrieve(query, top_k=top_k, use_graph=True)

        context_parts = []
        metadata_list = []
        for result in retrieval_results:
            context_parts.append(result["content"])
            metadata_list.append({
                "article": result["metadata"].get("article"),
                "score": result["final_score"],
                "source": result["source"]
            })

        context = "\n\n".join(context_parts)

        answer = self.generator.generate(query, context)

        span_results = self.detector.detect_hallucinations(answer, context)

        scores = self.scorer.get_scores_dict(span_results, context)

        hallucinated = self.detector.get_hallucinated_spans(span_results)

        result = {
            "query": query,
            "answer": answer,
            "retrieval_results": retrieval_results,
            "span_analysis": span_results,
            "hallucinated_spans": hallucinated,
            "scores": scores,
            "context": context
        }

        return result

    def build_graph(self, pdf_path: str):
        print(f"Building graph from {pdf_path}...")
        parser = LegalDocumentParser(config_path=self.config_path)
        chunks = parser.process(pdf_path=pdf_path)
        self.graph_builder.build(chunks)
        print("Graph built successfully")
