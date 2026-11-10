"""Identifier-occurrence pooling for paired PCA estimation."""
import torch


def paired_occurrence_differences(encoded, hidden):
    return torch.stack([
        hidden[0][left].mean(0) - hidden[1][right].mean(0)
        for left, right in zip(encoded[0].occurrences, encoded[1].occurrences)
    ])
