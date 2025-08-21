import glob
from dac_q4_its.utils.io import load_bin, save_bin
from dac_q4_its.quantization.int4_dynamic import dequantize_int4_per_row, delta_W
from dac_q4_its.adapters.build_adapters import build_adapters

def main():
    qfiles = sorted(glob.glob("artifacts/weights/layer_*_W_qint4.pkl"))
    for qf in qfiles:
        sf = qf.replace("W_qint4","scales")
        wf = qf.replace("W_qint4","W_fp32")
        vf = qf.replace("artifacts/weights","artifacts/eigs").replace("W_qint4.pkl","Vk.pkl")

        Q = load_bin(qf)
        S = load_bin(sf)
        Wfp = load_bin(wf)
        Vk = load_bin(vf)

        Wdq = dequantize_int4_per_row(Q, S)
        dW = delta_W(Wfp, Wdq)
        U, D = build_adapters(Vk, dW)
        save_bin(U, qf.replace("W_qint4","U"))
        save_bin(D, qf.replace("W_qint4","D"))
    print("Computed and saved adapters U,D per layer.")

if __name__ == "__main__":
    main()
