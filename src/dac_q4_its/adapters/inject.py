import torch
import torch.nn as nn

class CompressedLinear(nn.Module):
    """
    y = (dequant(Q) @ x) + U @ (D @ x)
    """
    def __init__(self, Q, scales, U, D):
        super().__init__()
        self.register_buffer("Q", Q)            # int8
        self.register_buffer("scales", scales)  # float
        self.register_buffer("U", U)            # float32
        self.register_buffer("D", D)            # float32

    def forward(self, x):  # x: [B, d]
        Wdq = (self.Q.float() * self.scales)    # [d, d]
        main = x @ Wdq.T                        # [B, d]
        adapt = (x @ self.D.T) @ self.U.T       # [B, d]
        return main + adapt
