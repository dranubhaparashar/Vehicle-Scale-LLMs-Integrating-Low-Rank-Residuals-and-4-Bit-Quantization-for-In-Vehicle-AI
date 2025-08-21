from pathlib import Path
from dac_q4_its.utils.seeds import set_seed
from dac_q4_its.data.build_corpus import write_corpus

def main():
    set_seed(7)
    Path("artifacts").mkdir(exist_ok=True)
    write_corpus("artifacts/calibration.txt", n=2000)
    print("Wrote artifacts/calibration.txt")

if __name__ == "__main__":
    main()
