import json, re
from typing import Tuple

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))

def soft_f1(pred: str, gold: str) -> float:
    p = set(normalize(pred).split())
    g = set(normalize(gold).split())
    inter = len(p & g)
    if not p or not g: return 0.0
    prec = inter/len(p); rec = inter/len(g)
    if prec+rec == 0: return 0.0
    return 2*prec*rec/(prec+rec)

def eval_file(jsonl_path: str) -> Tuple[float,float]:
    ems, f1s = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            pred = ex.get("pred",""); gold = ex.get("target","")
            ems.append(exact_match(pred, gold))
            f1s.append(soft_f1(pred, gold))
    n = max(1, len(ems))
    return sum(ems)/n, sum(f1s)/n
