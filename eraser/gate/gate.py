"""Native selective-policy network and gate checkpoint wrapper."""
import json
from pathlib import Path

import torch
from torch import nn


class SimpleGate(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.ln = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward_logits(self, features):
        return self.net(self.ln(features)).squeeze(-1)

    def forward(self, features):
        return self.forward_logits(features).sigmoid()


class GateModel:
    def __init__(self, model_dir, with_seed=None, device=None):
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        directory = Path(model_dir)
        if (directory / "gate.json").exists() and with_seed is None:
            self.cfg = json.loads((directory / "gate.json").read_text())
            weights = directory / "gate.pt"
            in_dim = self.cfg["dimension"] * 6 + 1
        else:
            suffix = f"_{with_seed}" if with_seed is not None else ""
            self.cfg = torch.load(directory / f"cfg{suffix}.pt", map_location="cpu", weights_only=True)
            weights = directory / f"gate_model{suffix}.pt"
            in_dim = self.cfg["in_dim"]
        self.threshold = self.cfg["threshold"]
        self.model = SimpleGate(in_dim, self.cfg["hidden"], self.cfg["dropout"]).to(self.device)
        self.model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
        self.model.eval()

    @torch.inference_mode()
    def score(self, feat):
        return self.model(feat.to(device=self.device, dtype=torch.float32).unsqueeze(0)).item()

    def predict(self, feat):
        return float(self.score(feat) > self.threshold)
