"""Validate the generated CSVs and build the final submission ZIP.

  python package.py --out ../output --zip ../your-submission.zip [--demo ../demo/demo.mp4]
Never includes .env, caches or virtualenvs.
"""
import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".cache", ".venv", "venv", ".git"}
SKIP_FILES = {".env"}
MAX_MB = 15


def check_csvs(out: Path) -> list[str]:
    problems = []
    for name, col, n in (("level1.csv", "json", 10), ("level2.csv", "answer", 120)):
        p = out / name
        if not p.exists():
            problems.append(f"missing {p}")
            continue
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != n:
            problems.append(f"{name}: expected {n} rows, found {len(rows)}")
        for r in rows:
            if not (r.get(col) or "").strip():
                problems.append(f"{name}: empty {col} for {r.get('bill_id')} {r.get('question_id', '')}")
            elif name == "level1.csv":
                try:
                    json.loads(r[col])
                except json.JSONDecodeError:
                    problems.append(f"level1.csv: invalid JSON for {r['bill_id']}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../output")
    ap.add_argument("--zip", default="../your-submission.zip")
    ap.add_argument("--demo", default=None)
    a = ap.parse_args()
    out, src, target = Path(a.out), Path(__file__).parent, Path(a.zip)
    problems = check_csvs(out)
    for p in problems:
        print("PROBLEM:", p)
    if problems:
        print("Fix the problems above before submitting.")
        return 1
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("level1.csv", "level2.csv"):
            z.write(out / f, f"output/{f}")
        for p in sorted(src.rglob("*")):
            if p.is_dir() or SKIP_DIRS & set(p.parts) or p.name in SKIP_FILES or p.suffix == ".pyc":
                continue
            z.write(p, f"source/{p.relative_to(src).as_posix()}")
        if a.demo and Path(a.demo).exists():
            z.write(a.demo, f"demo/{Path(a.demo).name}")
    mb = target.stat().st_size / 1e6
    print(f"Built {target} ({mb:.2f} MB)", "OK" if mb <= MAX_MB else "TOO BIG (limit 15 MB)")
    return 0 if mb <= MAX_MB else 1


if __name__ == "__main__":
    sys.exit(main())
