import torch

@torch.no_grad()
def topk_eigvecs_from_activations(A: torch.Tensor, k: int):
    """
    A: [N, d] activations for one layer across corpus tokens.
    returns V_k: [d, k]
    """
    C = A.T @ A
    eigvals, eigvecs = torch.linalg.eigh(C)
    V = eigvecs[:, -k:]
    return V
