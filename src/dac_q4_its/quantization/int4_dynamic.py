import torch

def per_row_scales(W: torch.Tensor, max_q=7.0):
    # W: [out, in]
    max_abs = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = max_abs / max_q
    return scales

def quantize_int4_per_row(W: torch.Tensor, qmin=-8, qmax=7):
    scales = per_row_scales(W, max_q=float(qmax))
    Q = torch.round(W / scales).clamp(qmin, qmax).to(torch.int8)
    return Q, scales

def dequantize_int4_per_row(Q: torch.Tensor, scales: torch.Tensor):
    return Q.float() * scales

def delta_W(W_fp: torch.Tensor, Wdq: torch.Tensor):
    return W_fp - Wdq
