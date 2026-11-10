"""Identical identifier/context features for training and inference."""
import torch
from torch.nn import functional as F


def gate_feature(hidden, identifiers, context):
    ident, ctx = hidden[identifiers].float(), hidden[context].float()
    mean_i, mean_c = ident.mean(0), ctx.mean(0)
    cosine = F.cosine_similarity(mean_i[None], mean_c[None], dim=-1)
    return torch.cat((mean_i, ident.max(0).values, mean_c, ctx.max(0).values,
                      (mean_i - mean_c).abs(), mean_i * mean_c, cosine))
