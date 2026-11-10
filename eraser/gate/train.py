"""Selective policy training and threshold calibration."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .gate import SimpleGate
from ..captures.act_captures import capture_features
from ..models.llm_wrapper import set_seed

@torch.inference_mode()
def calibrate(gate, features, device, target_fpr, batch_size):
    gate.eval()
    scores = []
    for group in features:
        scores.append(torch.cat([gate(batch.to(device)).cpu() for batch in group.split(batch_size)]))
    threshold = torch.quantile(scores[0], 1.0 - target_fpr).item()
    return {
        "threshold": threshold,
        "fpr": (scores[0] > threshold).float().mean().item(),
        "tpr": (scores[1] > threshold).float().mean().item(),
    }


def fit_gate(model, training, validation):
    config = model.config["gate"]
    if config["optimizer"] != "Adam" or not 0 < config["target_fpr"] < 1:
        raise ValueError("Expected Adam optimizer and 0 < target_fpr < 1")
    if config["epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("Gate epochs and batch_size must be positive")
    set_seed(model.config["seed"])
    train = capture_features(model, training, "Gate training features")
    valid = capture_features(model, validation, "Gate validation features")
    gate = SimpleGate(model.dimension * 6 + 1, config["hidden"], config["dropout"]).to(model.device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=config["lr"])
    loader = DataLoader(TensorDataset(*train), batch_size=config["batch_size"], shuffle=True)
    best = None
    history = []
    for epoch in range(config["epochs"]):
        gate.train()
        for clean, renamed in loader:
            clean_logits = gate.forward_logits(clean.to(model.device))
            renamed_logits = gate.forward_logits(renamed.to(model.device))
            p0, p1 = clean_logits.sigmoid(), renamed_logits.sigmoid()
            loss = config["clean_weight"] * nn.functional.binary_cross_entropy_with_logits(clean_logits, torch.zeros_like(clean_logits))
            loss += config["obfuscated_weight"] * nn.functional.binary_cross_entropy_with_logits(renamed_logits, torch.ones_like(renamed_logits))
            loss += config["binary_weight"] * ((p0 * (1 - p0)).mean() + (p1 * (1 - p1)).mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = calibrate(gate, valid, model.device, config["target_fpr"], config["batch_size"])
        history.append({"epoch": epoch + 1, **metrics})
        print(history[-1], flush=True)
        if best is None or metrics["tpr"] > best["tpr"]:
            best = {**metrics, "epoch": epoch + 1,
                    "state": {key: value.detach().cpu().clone() for key, value in gate.state_dict().items()}}
    best.update(training_pairs=len(train[0]), validation_pairs=len(valid[0]))
    return best, history
