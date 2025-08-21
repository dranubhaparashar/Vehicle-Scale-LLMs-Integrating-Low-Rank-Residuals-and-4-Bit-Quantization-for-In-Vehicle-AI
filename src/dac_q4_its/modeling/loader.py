import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ModelCfg:
    hidden_size: int = 256
    n_layers: int = 4
    vocab_size: int = 32000

class ToyTransformer(nn.Module):
    """
    Minimal model exposing projection matrices like a transformer:
    per-layer linear W (acts like Q/K/V/FFN all-in-one for demo).
    """
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
                                     for _ in range(cfg.n_layers)])
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

    def forward(self, x):  # x: token ids [B, T]
        h = self.embed(x).mean(dim=1)  # [B, H] crude pooling
        for lin in self.layers:
            h = lin(h)
            h = torch.relu(h)
        return h  # [B, H]

def load_toy(cfg: dict) -> ToyTransformer:
    return ToyTransformer(ModelCfg(**cfg))
