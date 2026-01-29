from dataclasses import dataclass
from typing import Iterable, List

from sentence_transformers import CrossEncoder


@dataclass
class RerankResult:
    doc_id: str
    score: float


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, points: Iterable, top_k: int = 10, payload_key: str = "text") -> List[RerankResult]:
        candidates = []
        for point in points:
            payload = point.payload or {}
            doc_text = payload.get(payload_key)
            if doc_text:
                candidates.append((str(point.id), doc_text))
            if len(candidates) >= top_k:
                break

        if not candidates:
            return []

        pairs = [(query, doc_text) for _, doc_text in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        ranked = sorted(
            (RerankResult(doc_id=doc_id, score=float(score)) for (doc_id, _), score in zip(candidates, scores)),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked
