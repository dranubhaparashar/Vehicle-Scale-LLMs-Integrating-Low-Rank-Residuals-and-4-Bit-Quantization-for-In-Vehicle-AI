import yaml, torch
from pathlib import Path
from dac_q4_its.adapters.eigenspace import topk_eigvecs_from_activations
from dac_q4_its.modeling.loader import load_toy
from dac_q4_its.utils.io import save_bin
from dac_q4_its.utils.seeds import set_seed

def main():
    set_seed(42)
    mcfg = yaml.safe_load(open("configs/model/llama-1b.yml"))
    rank = yaml.safe_load(open("configs/adapter/rank8_zero_init.yml"))["rank"]
    model = load_toy(mcfg).eval()

    B, T = 32, 8
    tok = torch.randint(0, mcfg["vocab_size"], (B, T))
    emb = model.embed(tok).mean(dim=1)  # [B, H]
    A_prev = emb
    Path("artifacts/eigs").mkdir(parents=True, exist_ok=True)
    for li, lin in enumerate(model.layers):
        h = lin(A_prev).relu()
        A_layer = h.detach()  # [B, H]
        Vk = topk_eigvecs_from_activations(A_layer, rank)  # [d, k]
        save_bin(Vk, f"artifacts/eigs/layer_{li}_Vk.pkl")
        A_prev = h
    print("Saved layer-wise eigenvectors (top-k).")

if __name__ == "__main__":
    main()
