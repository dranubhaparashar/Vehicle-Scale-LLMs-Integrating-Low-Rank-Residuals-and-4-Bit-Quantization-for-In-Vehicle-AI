#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_synthetic_utterances.py
--------------------------------
Deterministic generator for long-tail navigation utterances with rules for:
- Time (rush hour, arrive-before, depart-after)
- Weather (heavy rain, snow, fog)
- Detours/constraints (avoid tolls, avoid construction, scenic)
- Traffic (minimal congestion, avoid gridlock near event venues)

Outputs JSONL with fields:
  - input: natural language utterance
  - target: normalized "DEST=<token>,CONSTRAINT=<c1|c2|...>"

Reproducibility:
  - Fully deterministic given --seed and lexicons.
  - Duplicate prevention via a set keyed on (dest, constraint tokens, template id).

Usage:
  python scripts/generate_synthetic_utterances.py --out synthetic_nln_25k.jsonl --n 25000 --seed 2025
"""
import argparse, random, itertools, os, sys, json
from typing import List, Tuple, Dict
try:
    import yaml
except ImportError:
    yaml = None

# ---------------------------
# Lexicons (defaults)
# ---------------------------
DEFAULT_LEX = {
    "dests": [
        "airport","downtown","central_station","city_hospital","university",
        "tech_park","stadium","old_town","riverside","industrial_area"
    ],
    "roads": ["NH-48","Ring Road","Outer Ring Road","I-80","Maple Street","MG Road","Expressway 1"],
    "scenic": ["city_park","lakeside","botanical_garden","hill_road"],
    "events": ["cricket_stadium","concert_arena","Convention_Center"]
}

def load_lexicons(path: str) -> Dict[str, List[str]]:
    lex = DEFAULT_LEX.copy()
    if path and yaml and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            if isinstance(v, list) and v:
                lex[k] = v
    return lex

# ---------------------------
# Helpers
# ---------------------------
def canonical(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

def join_constraints(tokens: List[str]) -> str:
    return "|".join(tokens)

# ---------------------------
# Rule families
# ---------------------------
def rules_time(rng: random.Random) -> List[Tuple[str, List[str]]]:
    # returns (phrase, [tokens])
    time_rules = [
        ("during rush hour", ["rush_hour"]),
        ("before 8 am", ["arrive_before:08:00"]),
        ("before 9 am", ["arrive_before:09:00"]),
        ("after 7 pm", ["depart_after:19:00"]),
        ("after 8 pm", ["depart_after:20:00"]),
    ]
    return time_rules

def rules_weather(rng: random.Random) -> List[Tuple[str, List[str]]]:
    weather_rules = [
        ("in heavy rain", ["weather:heavy_rain","avoid_flooded"]),
        ("with dense fog", ["weather:fog","low_visibility"]),
        ("during snowfall", ["weather:snow","slippery_roads"]),
        ("in heatwave conditions", ["weather:heatwave"]),
    ]
    return weather_rules

def rules_detour(rng: random.Random, lex: Dict[str, List[str]]) -> List[Tuple[str, List[str]]]:
    roads = lex["roads"]
    scenic = lex["scenic"]
    # pick a few deterministic samples from roads/scenic via slicing
    detour_rules = [
        ("avoiding tolls", ["avoid_tolls"]),
        (f"avoiding construction on {roads[4]}", [f"avoid_construction:{canonical(roads[4])}"]),
        ("taking a scenic route", ["prefer_scenic"]),
        (f"passing through {scenic[0]}", [f"via:{canonical(scenic[0])}"]),
        (f"taking the next exit toward {roads[0]}", [f"next_exit:{canonical(roads[0])}"]),
        ("avoiding highways", ["avoid_highways"]),
        ("prefer highways", ["prefer_highways"]),
    ]
    return detour_rules

def rules_traffic(rng: random.Random, lex: Dict[str, List[str]]) -> List[Tuple[str, List[str]]]:
    events = lex["events"]
    roads = lex["roads"]
    traffic_rules = [
        ("with minimal traffic", ["min_traffic"]),
        (f"avoiding gridlock near the {events[0]}", [f"avoid_gridlock:{canonical(events[0])}"]),
        (f"avoiding congestion on {roads[0]}", [f"avoid_congestion:{canonical(roads[0])}"]),
        ("taking the fastest route", ["prefer_fastest"]),
    ]
    return traffic_rules

# ---------------------------
# Templates
# ---------------------------
BASE_TEMPLATES = [
    "navigate to {dest} {mods}",
    "find a route to {dest} {mods}",
    "get me to {dest} {mods}",
    "plan a trip to {dest} {mods}",
    "route to {dest} {mods}",
]

def render_sentence(template_id: int, dest: str, modifiers: List[str]) -> str:
    # natural conjunctions
    if not modifiers:
        mods = ""
    elif len(modifiers) == 1:
        mods = modifiers[0]
    else:
        mods = ", ".join(modifiers[:-1]) + " and " + modifiers[-1]
    s = BASE_TEMPLATES[template_id].format(dest=dest.replace("_"," "), mods=mods).strip()
    s = " ".join(s.split())  # normalize spaces
    return s

# ---------------------------
# Main generation
# ---------------------------
def generate(n: int, seed: int, lex: Dict[str, List[str]], compose: bool, max_constraints: int):
    rng = random.Random(seed)
    dests = lex["dests"]
    TR = rules_time(rng)
    WR = rules_weather(rng)
    DR = rules_detour(rng, lex)
    AR = rules_traffic(rng, lex)

    # pool of single-rule phrases
    singles = []
    for fam, rules in enumerate([TR, WR, DR, AR]):
        for phrase, toks in rules:
            singles.append((fam, phrase, toks))

    # Build combinations deterministically
    outputs = []
    seen = set()
    t_count = len(BASE_TEMPLATES)
    d_count = len(dests)
    s_count = len(singles)

    # Round-robin over template, destination, and constraints
    i = 0
    while len(outputs) < n:
        tid = i % t_count
        dest = dests[i % d_count]
        # choose 1..max_constraints distinct rules
        if compose and max_constraints > 1:
            k = 1 + (i // (t_count*d_count)) % max_constraints
        else:
            k = 1
        # pick k rules via deterministic offsets
        idxs = []
        for j in range(k):
            idxs.append((i + j*7) % s_count)  # stride 7 for mixing
        chosen = [singles[j] for j in idxs]
        phrases = [c[1] for c in chosen]
        tokens = []
        for c in chosen:
            for tok in c[2]:
                if tok not in tokens:
                    tokens.append(tok)

        # key to avoid duplicates
        key = (tid, dest, tuple(tokens))
        if key in seen:
            i += 1
            continue
        seen.add(key)

        sent = render_sentence(tid, dest, phrases).strip()
        target = f"DEST={canonical(dest)},CONSTRAINT={join_constraints(tokens)}"
        outputs.append({"input": sent, "target": target})
        i += 1

    return outputs[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="synthetic_nln_25k.jsonl")
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--lexicons", type=str, default="data/lexicons.yaml")
    ap.add_argument("--compose", action="store_true", help="enable multi-constraint compositions")
    ap.add_argument("--no-compose", dest="compose", action="store_false")
    ap.set_defaults(compose=True)
    ap.add_argument("--max-constraints", type=int, default=3)
    args = ap.parse_args()

    lex = load_lexicons(args.lexicons)
    data = generate(args.n, args.seed, lex, args.compose, args.max_constraints)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(data)} lines to {args.out}")

if __name__ == "__main__":
    main()
