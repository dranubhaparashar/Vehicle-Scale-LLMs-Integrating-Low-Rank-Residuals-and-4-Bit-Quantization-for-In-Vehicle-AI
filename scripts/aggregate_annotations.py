#!/usr/bin/env python3
# aggregate_annotations.py
# Merge multiple per-annotator CSV files and sanity-check ranges.

import argparse, pandas as pd

COLS = ["item_id","annotator_id","prompt","response","coherence","relevance","instruction_following","safety_compliance","fluency","notes"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="list of CSV files")
    ap.add_argument("--out", default="annotations_merged.csv")
    args = ap.parse_args()

    dfs = []
    for p in args.inputs:
        df = pd.read_csv(p)
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{p} missing columns: {missing}")
        dfs.append(df[COLS])
    out = pd.concat(dfs, ignore_index=True)
    for c in ["coherence","relevance","instruction_following","safety_compliance","fluency"]:
        bad = out[~out[c].between(1,5)]
        if len(bad):
            raise ValueError(f"Found out-of-range values in {c}")
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(out)} rows.")

if __name__ == "__main__":
    main()
