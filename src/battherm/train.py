"""Generic PyTorch training loop with early stopping on validation RMSE."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass
class Arrays:
    """Model-ready, already scaled arrays. `base` is added to the network output
    (zeros for plain nets, the scaled physics forecast for the hybrid model)."""

    past: np.ndarray
    future: np.ndarray
    y: np.ndarray
    base: np.ndarray

    def tensors(self) -> list[torch.Tensor]:
        return [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in (self.past, self.future, self.y, self.base)]


def _loader(a: Arrays, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(*a.tensors()), batch_size=batch_size, shuffle=shuffle, generator=g)


@torch.no_grad()
def predict(model: nn.Module, a: Arrays, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for past, future, _, base in _loader(a, batch_size, False, 0):
        out.append((model(past.to(device), future.to(device)) + base.to(device)).cpu().numpy())
    return np.concatenate(out)


def fit(model: nn.Module, train: Arrays, val: Arrays, cfg: dict, device: torch.device, seed: int, log=print) -> dict:
    """Train with Adam + MSE; keep the weights with the best validation RMSE."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    loss_fn = nn.MSELoss()
    loader = _loader(train, cfg["batch_size"], True, seed)
    # Optional: halve the learning rate when validation RMSE stalls for `lr_plateau_patience` epochs
    sched = None
    if cfg.get("lr_plateau", False):
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=cfg.get("lr_plateau_patience", 3))
    history = {"epoch": [], "train_rmse": [], "val_rmse": [], "seconds": [], "lr": []}
    best_rmse, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, sq_sum, n = time.time(), 0.0, 0
        for past, future, y, base in loader:
            past, future, y, base = past.to(device), future.to(device), y.to(device), base.to(device)
            loss = loss_fn(model(past, future) + base, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            opt.step()
            sq_sum += loss.item() * len(y)
            n += len(y)
        val_rmse = float(np.sqrt(np.mean((predict(model, val, device) - val.y) ** 2)))
        history["epoch"].append(epoch)
        history["train_rmse"].append(float(np.sqrt(sq_sum / n)))
        history["val_rmse"].append(val_rmse)
        history["seconds"].append(time.time() - t0)
        history["lr"].append(opt.param_groups[0]["lr"])
        if sched is not None:
            sched.step(val_rmse)
        log(f"  epoch {epoch:3d}  train {history['train_rmse'][-1]:.4f}  val {val_rmse:.4f}  (scaled units)")

        if val_rmse < best_rmse - 1e-5:
            best_rmse, best_state, bad_epochs = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                log(f"  early stop at epoch {epoch}; best val {best_rmse:.4f}")
                break
    model.load_state_dict(best_state)
    history["best_val_rmse"] = best_rmse
    return history
