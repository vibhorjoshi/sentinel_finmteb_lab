import numpy as np
import torch
from scipy.stats import ortho_group
from sentence_transformers import SentenceTransformer


class SentinelEmbedder:
    def __init__(self, device=None, verbose=True, vector_dim=1536):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        if self.verbose:
            print(f"--- 🚀 Initializing Qwen-2.5 Core on {self.device.upper()} ---")

        self.model = SentenceTransformer(
            "Alibaba-NLP/gte-Qwen2-1.5b-instruct",
            device=self.device,
            trust_remote_code=True,
        )

        self.model._first_module().auto_model.config.use_cache = False

        P_raw = ortho_group.rvs(dim=vector_dim)
        self.P_matrix = torch.tensor(P_raw, dtype=torch.float32).to(self.device)

    def encode(
        self,
        texts,
        batch_size=64,
        show_progress_bar=False,
        persona="Forensic Auditor",
        normalize_embeddings=True,
    ):
        augmented = [f"System: [Persona: {persona}] | Content: {t}" for t in texts]

        with torch.no_grad():
            embeddings = self.model.encode(
                augmented,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=show_progress_bar,
            )
            rotated = torch.matmul(embeddings, self.P_matrix)
            if normalize_embeddings:
                rotated = torch.nn.functional.normalize(rotated, p=2, dim=1)

        return rotated.cpu().numpy()


class FinancialEmbedder(SentinelEmbedder):
    pass


class QwenFinancialEmbedder(SentinelEmbedder):
    pass
