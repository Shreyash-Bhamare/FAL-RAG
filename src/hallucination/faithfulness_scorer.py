# src/hallucination/faithfulness_scorer.py

from typing import List, Dict


class FaithfulnessScorer:

    def __init__(self, weights: dict = None):
        if weights is None:
            weights = {
                "faithfulness_score": 0.5,
                "unsupported_claim_score": 0.3,
                "evidence_coverage_score": 0.2
            }
        self.weights = weights

    def calculate_faithfulness_score(self, spans_with_status: List[Dict]) -> float:
        if not spans_with_status:
            return 0.0

        supported_count = sum(1 for s in spans_with_status if s["status"] == "supported")
        total_count = len(spans_with_status)

        return supported_count / total_count if total_count > 0 else 0.0

    def calculate_unsupported_claim_score(self, spans_with_status: List[Dict]) -> float:
        if not spans_with_status:
            return 0.0

        hallucinated_count = sum(1 for s in spans_with_status if s["status"] == "hallucinated")
        total_count = len(spans_with_status)

        unsupported_ratio = hallucinated_count / total_count if total_count > 0 else 0.0
        return 1.0 - unsupported_ratio

    def calculate_evidence_coverage_score(self, spans_with_status: List[Dict], context: str) -> float:
        if not spans_with_status or not context:
            return 0.0

        supported_confidence_sum = sum(
            s["confidence"] for s in spans_with_status 
            if s["status"] in ["supported", "uncertain"]
        )

        avg_confidence = supported_confidence_sum / len(spans_with_status) if spans_with_status else 0.0

        return avg_confidence

    def calculate_overall_faithfulness(self, spans_with_status: List[Dict], context: str) -> float:
        faithfulness = self.calculate_faithfulness_score(spans_with_status)
        unsupported = self.calculate_unsupported_claim_score(spans_with_status)
        coverage = self.calculate_evidence_coverage_score(spans_with_status, context)

        overall = (
            faithfulness * self.weights["faithfulness_score"] +
            unsupported * self.weights["unsupported_claim_score"] +
            coverage * self.weights["evidence_coverage_score"]
        )

        return round(overall, 4)

    def get_scores_dict(self, spans_with_status: List[Dict], context: str) -> Dict:
        return {
            "faithfulness_score": round(self.calculate_faithfulness_score(spans_with_status), 4),
            "unsupported_claim_score": round(self.calculate_unsupported_claim_score(spans_with_status), 4),
            "evidence_coverage_score": round(self.calculate_evidence_coverage_score(spans_with_status, context), 4),
            "overall_faithfulness": self.calculate_overall_faithfulness(spans_with_status, context)
        }
