"""
Parse the plain-text "Speedup Summary" + per-variant blocks emitted by each
C++ binary (svm/knn/mlp/dt/nb) into rows matching the results.csv schema.

Each binary prints three [<tag>] blocks — Serial / OpenMP / pthreads — each
followed by timing + metrics lines. See nb/nb.cpp:652-666 for the shared
print_results format; all five algorithms match.

Usage:
    python parse_cpp_output.py logs/svm.out logs/knn.out ... --append results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


CSV_COLUMNS = [
    "algorithm",
    "framework",
    "variant",
    "n_threads",
    "train_ms",
    "infer_ms",
    "acc",
    "prec",
    "rec",
    "f1",
    "notes",
]

# Pattern: "[Serial SVM]" or "[Parallel SVM (OpenMP, 8 threads)]"
#          or "[Parallel SVM (pthreads, 8 threads)]"
BLOCK_HEADER = re.compile(
    r"^\[(?P<kind>Serial|Parallel)\s+(?P<algo>[A-Z]+)"
    r"(?:\s+\((?P<engine>OpenMP|pthreads),\s*(?P<threads>\d+)\s*threads\))?\]"
)

METRIC_PATTERNS = {
    "train_ms": re.compile(r"^\s+Training time\s+:\s+([\d.]+)\s+ms"),
    "infer_ms": re.compile(r"^\s+Inference time\s+:\s+([\d.]+)\s+ms"),
    "acc":      re.compile(r"^\s+Accuracy\s+:\s+([\d.]+)"),
    "prec":     re.compile(r"^\s+Precision\s+:\s+([\d.]+)"),
    "rec":      re.compile(r"^\s+Recall\s+:\s+([\d.]+)"),
    "f1":       re.compile(r"^\s+F1 Score\s+:\s+([\d.]+)"),
}


def parse_log(path: Path):
    """Yield dict rows for the three variants in one binary's stdout log."""
    lines = path.read_text().splitlines()
    current = None
    rows = []
    for line in lines:
        m = BLOCK_HEADER.match(line)
        if m:
            if current is not None:
                rows.append(current)
            current = {
                "kind": m.group("kind"),
                "algo": m.group("algo"),
                "engine": m.group("engine"),
                "threads": int(m.group("threads")) if m.group("threads") else 1,
                "train_ms": None,
                "infer_ms": None,
                "acc": None,
                "prec": None,
                "rec": None,
                "f1": None,
            }
            continue
        if current is None:
            continue
        for key, pat in METRIC_PATTERNS.items():
            mm = pat.match(line)
            if mm:
                current[key] = float(mm.group(1))
                break
    if current is not None:
        rows.append(current)

    output = []
    for r in rows:
        variant = "serial" if r["kind"] == "Serial" else (
            "omp" if r["engine"] == "OpenMP" else "pthreads"
        )
        notes = "parsed from C++ stdout"
        output.append({
            "algorithm": r["algo"].lower(),
            "framework": "ours-cpp",
            "variant": variant,
            "n_threads": r["threads"],
            "train_ms": round(r["train_ms"], 2) if r["train_ms"] is not None else 0.0,
            "infer_ms": round(r["infer_ms"], 2) if r["infer_ms"] is not None else 0.0,
            "acc": round(r["acc"], 4) if r["acc"] is not None else 0.0,
            "prec": round(r["prec"], 4) if r["prec"] is not None else 0.0,
            "rec": round(r["rec"], 4) if r["rec"] is not None else 0.0,
            "f1": round(r["f1"], 4) if r["f1"] is not None else 0.0,
            "notes": notes,
        })
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="C++ stdout log files")
    ap.add_argument("--append", default="results.csv",
                    help="CSV path to append to (created if absent)")
    ap.add_argument("--json", default="results.json",
                    help="Parallel JSON file to append to")
    args = ap.parse_args()

    rows = []
    for p in args.logs:
        path = Path(p)
        if not path.exists():
            print(f"warning: {path} does not exist - skipping")
            continue
        parsed = parse_log(path)
        if not parsed:
            print(f"warning: {path} produced 0 rows - format mismatch?")
        rows.extend(parsed)

    out_csv = Path(args.append)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    out_json = Path(args.json)
    existing = []
    if out_json.exists():
        try:
            existing = json.loads(out_json.read_text())
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []
    existing.extend(rows)
    out_json.write_text(json.dumps(existing, indent=2))

    print(f"Parsed {len(rows)} rows from {len(args.logs)} log(s); appended to {out_csv}")


if __name__ == "__main__":
    main()
