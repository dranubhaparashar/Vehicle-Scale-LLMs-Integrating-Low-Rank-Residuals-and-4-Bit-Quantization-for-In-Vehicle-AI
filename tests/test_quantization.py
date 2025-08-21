import torch
from dac_q4_its.quantization.int4_dynamic import quantize_int4_per_row, dequantize_int4_per_row

def test_qdq_roundtrip():
    W = torch.randn(32, 32)
    Q, S = quantize_int4_per_row(W)
    Wdq = dequantize_int4_per_row(Q, S)
    err = (W - Wdq).abs().mean().item()
    assert err < 0.5  # coarse bound for 4-bit demo
