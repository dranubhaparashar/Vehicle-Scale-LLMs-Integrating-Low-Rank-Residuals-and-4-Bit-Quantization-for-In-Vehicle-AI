import json
from dac_q4_its.evaluation.nln import eval_file

def main():
    preds = [
      {"input":"navigate to airport avoiding tolls","pred":"DEST=airport,CONSTRAINT=avoid_tolls","target":"DEST=airport,CONSTRAINT=avoid_tolls"},
      {"input":"find a route to downtown with minimal traffic","pred":"DEST=downtown,CONSTRAINT=min_traffic","target":"DEST=downtown,CONSTRAINT=min_traffic"}
    ]
    with open("artifacts/nln_preds.jsonl","w",encoding="utf-8") as f:
        for p in preds: f.write(json.dumps(p)+"\n")
    em, f1 = eval_file("artifacts/nln_preds.jsonl")
    print(f"NLN EM: {em:.3f}, SOFT-F1: {f1:.3f}")

if __name__ == "__main__":
    main()
