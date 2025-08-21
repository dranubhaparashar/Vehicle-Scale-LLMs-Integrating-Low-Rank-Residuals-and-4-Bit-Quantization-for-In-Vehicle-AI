import torch

@torch.no_grad()
def build_adapters(Vk: torch.Tensor, dW: torch.Tensor):
    """
    Vk: [d, k], dW: [out=d, in=d]
    U = Vk, D = Vk^T dW  (rank-k)
    """
    U = Vk          # [d, k]
    D = Vk.T @ dW   # [k, d]
    return U.contiguous(), D.contiguous()
