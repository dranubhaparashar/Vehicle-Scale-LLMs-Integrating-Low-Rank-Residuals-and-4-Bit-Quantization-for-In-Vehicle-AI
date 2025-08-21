import glob, yaml
from dac_q4_its.quantization.int4_dynamic import quantize_int4_per_row
from dac_q4_its.utils.io import load_bin, save_bin

def main():
    qcfg = yaml.safe_load(open("configs/quant/int4_dynamic.yml"))
    paths = sorted(glob.glob("artifacts/weights/layer_*_W_fp32.pkl"))
    for p in paths:
        W = load_bin(p)  # torch tensor [d, d]
        Q, scales = quantize_int4_per_row(W, qmin=qcfg["range_min"], qmax=qcfg["range_max"])
        save_bin(Q, p.replace("W_fp32","W_qint4"))
        save_bin(scales, p.replace("W_fp32","scales"))
    print("Quantized INT4 weights saved.")

if __name__ == "__main__":
    main()
