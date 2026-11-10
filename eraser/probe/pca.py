"""PCA direction estimation."""
import torch
from tqdm import tqdm
from ..erasers.pca_support import paired_occurrence_differences

class PCAStatistics:
    """Centered PCA via running sufficient statistics; no full activation cache."""
    def __init__(self, dimension):
        self.count = 0
        self.mean = torch.zeros(dimension, dtype=torch.float64)
        self.scatter = torch.zeros((dimension, dimension), dtype=torch.float64)

    def update(self, rows):
        rows = rows.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(rows).all():
            raise ValueError("Non-finite activation difference")
        count = len(rows)
        if not count:
            return
        mean = rows.mean(0)
        centered = rows - mean
        delta = mean - self.mean
        total = self.count + count
        self.scatter += centered.T @ centered + torch.outer(delta, delta) * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total

    def basis(self, k):
        if not 0 < k <= min(len(self.mean), self.count - 1):
            raise ValueError(f"k={k} requires dimension >= k and at least k+1 paired occurrences; got {self.count}")
        _, vectors = torch.linalg.eigh(self.scatter / (self.count - 1))
        return vectors[:, -k:].flip(1).float().contiguous()


def fit_pca(model, rows):
    statistics = PCAStatistics(model.dimension)
    used = 0
    for row in tqdm(rows, desc="PCA pairs"):
        try:
            encoded = [model.encode(row[key], row.get("func_name", "")) for key in ("code", "vari_code")]
            if any(not item.identifiers or not item.context for item in encoded):
                continue
            if len(encoded[0].occurrences) != len(encoded[1].occurrences):
                continue
            hidden = [model.capture(item) for item in encoded]
            # Pair occurrences before pooling: identifier subword lengths may differ.
            differences = paired_occurrence_differences(encoded, hidden)
            statistics.update(differences)
            used += 1
        except (ValueError, KeyError, IndexError) as error:
            raise ValueError(f"PCA record {row.get('id', '?')}: {error}") from error
    return statistics.basis(model.config["k"]), statistics.count, used


