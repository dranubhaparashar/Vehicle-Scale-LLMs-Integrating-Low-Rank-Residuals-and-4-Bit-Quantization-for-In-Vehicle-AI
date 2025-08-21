import argparse, yaml, torch, json
from dac_q4_its.modeling.loader import load_toy
from dac_q4_its.adapters.inject import CompressedLinear
from dac_q4_its.utils.io import load_bin

def build_compressed_model(mcfg):
    model = load_toy(mcfg).eval()
    for li, lin in enumerate(model.layers):
        Q = load_bin(f"artifacts/weights/layer_{li}_W_qint4.pkl")
        S = load_bin(f"artifacts/weights/layer_{li}_scales.pkl")
        U = load_bin(f"artifacts/weights/layer_{li}_U.pkl")
        D = load_bin(f"artifacts/weights/layer_{li}_D.pkl")
        comp = CompressedLinear(Q, S, U, D)
        model.layers[li] = comp
    return model

def toy_decode(vec):
    return "DEST=airport,CONSTRAINT=avoid_tolls" if vec.mean() > 0 else "DEST=downtown,CONSTRAINT=min_traffic"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="navigate to airport avoiding tolls")
    args = parser.parse_args()

    mcfg = yaml.safe_load(open("configs/model/llama-1b.yml"))
    model = build_compressed_model(mcfg)
    tok = torch.randint(0, mcfg["vocab_size"], (1, 8))
    h = model(tok)
    pred = toy_decode(h.detach())
    print(json.dumps({"input": args.prompt, "pred": pred}, indent=2))

if __name__ == "__main__":
    main()
