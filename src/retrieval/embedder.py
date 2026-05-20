# src/retrieval/embedder.py

import torch
import numpy as np
from typing import List, Union
from transformers import AutoTokenizer, AutoModel
from src.utils.helpers import load_config


class InLegalBERTEmbedder:

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.model_name = self.config["embeddings"]["model"]
        self.max_length = self.config["embeddings"]["max_length"]
        self.batch_size = self.config["embeddings"]["batch_size"]

        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {'GPU ✅' if self.device.type == 'cuda' else 'CPU ⚙️'}")

        # Load model and tokenizer
        print(f"   Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"   ✅ InLegalBERT loaded successfully")

    # ═══════════════════════════════════════════════════════════════
    # MEAN POOLING
    # ═══════════════════════════════════════════════════════════════

    def _mean_pooling(
        self,
        model_output: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean pooling on token embeddings.
        Averages all token vectors weighted by attention mask.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    # ═══════════════════════════════════════════════════════════════
    # SINGLE TEXT EMBEDDING
    # ═══════════════════════════════════════════════════════════════

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.
        Returns a 768-dimensional vector as a Python list.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.model(**encoded)

        embedding = self._mean_pooling(output, encoded["attention_mask"])

        # Normalize to unit vector
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.squeeze().cpu().numpy().tolist()

    # ═══════════════════════════════════════════════════════════════
    # BATCH EMBEDDING
    # ═══════════════════════════════════════════════════════════════

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts in batches for efficiency.
        Returns a list of 768-dimensional vectors.
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts.")

        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(0, len(texts), self.batch_size):
            batch_num = (batch_idx // self.batch_size) + 1
            batch = texts[batch_idx: batch_idx + self.batch_size]

            # Tokenize batch
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                output = self.model(**encoded)

            embeddings = self._mean_pooling(output, encoded["attention_mask"])

            # Normalize to unit vectors
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Convert to Python list
            batch_embeddings = embeddings.cpu().numpy().tolist()
            all_embeddings.extend(batch_embeddings)

            print(f"   Batch {batch_num}/{total_batches} "
                  f"[{len(batch)} chunks] ✅")

        return all_embeddings

    # ═══════════════════════════════════════════════════════════════
    # EMBEDDING DIMENSION
    # ═══════════════════════════════════════════════════════════════

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension of the model."""
        test_embedding = self.embed_text("test")
        return len(test_embedding)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT — for standalone testing
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n📌 Testing InLegalBERT Embedder...")
    embedder = InLegalBERTEmbedder(config_path="config/config.yaml")

    # Test single embedding
    test_text = "Article 21 guarantees the right to life and personal liberty."
    embedding = embedder.embed_text(test_text)
    print(f"\n✅ Single embedding test:")
    print(f"   Text    : {test_text}")
    print(f"   Dim     : {len(embedding)}")
    print(f"   Sample  : {embedding[:5]}")

    # Test batch embedding
    test_batch = [
        "Article 14 guarantees equality before law.",
        "Article 19 provides freedom of speech.",
        "Article 32 provides right to constitutional remedies."
    ]
    batch_embeddings = embedder.embed_batch(test_batch)
    print(f"\n✅ Batch embedding test:")
    print(f"   Texts     : {len(test_batch)}")
    print(f"   Embeddings: {len(batch_embeddings)}")
    print(f"   Dim each  : {len(batch_embeddings[0])}")
