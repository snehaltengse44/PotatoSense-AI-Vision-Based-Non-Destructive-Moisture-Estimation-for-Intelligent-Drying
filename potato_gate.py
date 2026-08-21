import numpy as np


class PotatoGate:
    def __init__(self, reference_embeddings_path, threshold=0.65):
        self.reference_embeddings = np.load(reference_embeddings_path)  # (N, 2048), L2-normalized
        self.threshold = threshold

    def check(self, feature_vector: np.ndarray) -> dict:
        """feature_vector: raw (2048,) ResNet50 output, not yet normalized."""
        vec = feature_vector / (np.linalg.norm(feature_vector) + 1e-8)
        similarities = self.reference_embeddings @ vec
        best_sim = float(np.max(similarities))

        return {
            "is_valid": best_sim >= self.threshold,
            "similarity": best_sim,
        }