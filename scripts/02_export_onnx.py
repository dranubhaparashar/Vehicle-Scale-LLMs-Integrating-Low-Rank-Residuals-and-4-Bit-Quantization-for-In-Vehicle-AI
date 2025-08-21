import yaml, torch
from pathlib import Path
from dac_q4_its.modeling.loader import load_toy
from dac_q4_its.utils.io import save_bin

def main():
    cfg = yaml.safe_load(open("configs/model/llama-1b.yml"))
    model = load_toy(cfg)
    Path("artifacts/weights").mkdir(parents=True, exist_ok=True)
    for li, lin in enumerate(model.layers):
        W = lin.weight.data.to(torch.float32).clone()  # [d, d]
        save_bin(W, f"artifacts/weights/layer_{li}_W_fp32.pkl")
    print("Exported toy FP weights to artifacts/weights")

if __name__ == "__main__":
    main()
