# src/hallucination/span_detector.py

from typing import List, Dict
import re
from src.retrieval.embedder import InLegalBERTEmbedder
import numpy as np


class SpanHallucinationDetector:

    def __init__(self, embedder: InLegalBERTEmbedder, supported_threshold: float = 0.65, uncertain_threshold: float = 0.40):
        self.embedder = embedder
        self.supported_threshold = supported_threshold
        self.uncertain_threshold = uncertain_threshold
        self.generic_phrases = {"based on", "according to", "the provided", "the context", "stated above"}
        self.embedding_cache = {}

    def detect_hallucinations(self, answer: str, context: str) -> List[Dict]:
        spans = self._split_into_spans(answer)
        
        context_embedding = self._get_embedding(context)
        
        results = []
        for span in spans:
            if self._is_generic_phrase(span):
                continue

            span_embedding = self._get_embedding(span)
            confidence = self._cosine_similarity(span_embedding, context_embedding)
            status = self._classify_status(confidence)

            results.append({
                "span": span,
                "confidence": round(confidence, 4),
                "status": status
            })

        return results

    def _get_embedding(self, text: str) -> list:
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = self.embedder.embed_text(text)
        self.embedding_cache[text] = embedding
        return embedding

    def _split_into_spans(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        spans = []

        for sentence in sentences:
            words = sentence.split()
            if len(words) < 5:
                spans.append(sentence.strip())
            else:
                for i in range(len(words)):
                    for j in range(i + 4, min(i + 8, len(words) + 1)):
                        span = ' '.join(words[i:j])
                        spans.append(span)

        return list(set(spans))

    def _is_generic_phrase(self, span: str) -> bool:
        span_lower = span.lower()
        return any(phrase in span_lower for phrase in self.generic_phrases) or len(span.split()) < 3

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _classify_status(self, confidence: float) -> str:
        if confidence >= self.supported_threshold:
            return "supported"
        elif confidence >= self.uncertain_threshold:
            return "uncertain"
        else:
            return "hallucinated"

    def get_hallucinated_spans(self, spans_with_status: List[Dict]) -> List[Dict]:
        return [s for s in spans_with_status if s["status"] == "hallucinated"]

    def get_summary(self, spans_with_status: List[Dict]) -> Dict:
        if not spans_with_status:
            return {"total_spans": 0, "supported": 0, "uncertain": 0, "hallucinated": 0, "hallucination_rate": 0}

        total = len(spans_with_status)
        supported = sum(1 for s in spans_with_status if s["status"] == "supported")
        uncertain = sum(1 for s in spans_with_status if s["status"] == "uncertain")
        hallucinated = sum(1 for s in spans_with_status if s["status"] == "hallucinated")

        return {
            "total_spans": total,
            "supported": supported,
            "uncertain": uncertain,
            "hallucinated": hallucinated,
            "hallucination_rate": round(hallucinated / total, 4) if total > 0 else 0
        }