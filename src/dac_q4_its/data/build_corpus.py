from pathlib import Path
from .tokenization import simple_norm

TEMPLATE_NAV = [
    "navigate to {dest} avoiding tolls",
    "route to {dest} via highways",
    "find the fastest path to {dest}",
]
DESTS = ["airport", "downtown", "station", "office_park"]

def build_corpus(n=1000):
    lines = []
    for i in range(n):
        t = TEMPLATE_NAV[i % len(TEMPLATE_NAV)]
        d = DESTS[i % len(DESTS)]
        lines.append(simple_norm(t.format(dest=d)))
    return lines

def write_corpus(path: str, n=1000):
    lines = build_corpus(n)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
