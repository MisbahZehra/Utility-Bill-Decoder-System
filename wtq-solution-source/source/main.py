"""Utility Bill Decoder - CLI.

  python main.py run  --bills ../test/bills --level1 ../test/csv/level1.csv --level2 ../test/csv/level2.csv --out ../output
  python main.py demo --bill ../train/bills/KESC_0008.png --ask "How much of my bill is taxes?"
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from config import ConfigError, load_config
from imaging import load_image_b64
from level1 import empty_result, extract_bill
from level2 import answer_questions
from llm_client import LLMClient

log = logging.getLogger("main")
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def find_bill(bills_dir: Path, bill_id: str) -> Path | None:
    for ext in IMG_EXT:
        for cand in (bills_dir / f"{bill_id}{ext}", bills_dir / f"{bill_id}{ext.upper()}"):
            if cand.exists():
                return cand
    return None


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def cache_get(cache_dir: Path, name: str):
    p = cache_dir / name
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def cache_put(cache_dir: Path, name: str, data):
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def cmd_run(args) -> int:
    client = LLMClient(load_config())
    bills_dir, out_dir, cache_dir = Path(args.bills), Path(args.out), Path(args.cache)
    f1, rows1 = read_csv(Path(args.level1))
    f2, rows2 = read_csv(Path(args.level2))

    bill_ids = list(dict.fromkeys([r["bill_id"] for r in rows1] + [r["bill_id"] for r in rows2]))
    if args.limit:
        bill_ids = bill_ids[: args.limit]
    log.info("Processing %d bills", len(bill_ids))

    l1_by_bill: dict[str, dict] = {}
    l2_by_bill: dict[str, dict] = {}
    for i, bid in enumerate(bill_ids, 1):
        log.info("[%d/%d] %s", i, len(bill_ids), bid)
        questions = [(r["question_id"], r["question"]) for r in rows2 if r["bill_id"] == bid]
        path = find_bill(bills_dir, bid)
        image = None
        if path is None:
            log.error("%s: image not found in %s", bid, bills_dir)
        else:
            try:
                image = load_image_b64(str(path))
            except Exception as e:  # unreadable image -> graceful fallback
                log.error("%s: cannot read image: %s", bid, e)

        l1 = cache_get(cache_dir, f"{bid}.l1.json")
        if l1 is None:
            l1 = extract_bill(client, bid, image) if image else empty_result()
            if image:
                cache_put(cache_dir, f"{bid}.l1.json", l1)
        l1_by_bill[bid] = l1

        l2 = cache_get(cache_dir, f"{bid}.l2.json")
        if l2 is None or any(qid not in l2 for qid, _ in questions):
            if image:
                l2 = answer_questions(client, bid, image, l1, questions)
                cache_put(cache_dir, f"{bid}.l2.json", l2)
            else:
                l2 = {qid: "The bill image could not be read, so this question cannot be answered." for qid, _ in questions}
        l2_by_bill[bid] = l2

    for r in rows1:
        r["json"] = json.dumps(l1_by_bill.get(r["bill_id"], empty_result()), ensure_ascii=False, separators=(",", ":"))
    for r in rows2:
        r["answer"] = l2_by_bill.get(r["bill_id"], {}).get(r["question_id"], "") or \
            "This could not be determined from the bill."
    write_csv(out_dir / "level1.csv", f1, rows1)
    write_csv(out_dir / "level2.csv", f2, rows2)
    log.info("Wrote %s and %s (%d + %d rows)", out_dir / "level1.csv", out_dir / "level2.csv", len(rows1), len(rows2))
    return 0


def cmd_demo(args) -> int:
    client = LLMClient(load_config())
    path = Path(args.bill)
    image = load_image_b64(str(path))
    print(f"\n=== Bill: {path.name} ===\n--- Level 1: extracted fields ---")
    l1 = extract_bill(client, path.stem, image)
    print(json.dumps(l1, indent=2))
    qs = args.ask or ["How much of my bill is taxes?", "How does this month's usage compare with my previous months?",
                      "How much extra would I pay if I miss the due date?"]
    print("\n--- Level 2: answers ---")
    ans = answer_questions(client, path.stem, image, l1, [(f"Q{i+1}", q) for i, q in enumerate(qs)])
    for i, q in enumerate(qs):
        print(f"\nQ: {q}\nA: {ans[f'Q{i+1}']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Utility Bill Decoder")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process a folder of bills and fill the two CSV templates")
    r.add_argument("--bills", required=True, help="folder with bill images")
    r.add_argument("--level1", required=True, help="level1.csv template")
    r.add_argument("--level2", required=True, help="level2.csv template")
    r.add_argument("--out", default="output", help="output folder")
    r.add_argument("--cache", default=".cache", help="folder for resumable per-bill results")
    r.add_argument("--limit", type=int, default=0, help="only process the first N bills (testing)")
    d = sub.add_parser("demo", help="process a single bill and print the results")
    d.add_argument("--bill", required=True)
    d.add_argument("--ask", action="append", help="question to ask (repeatable)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    try:
        return cmd_run(args) if args.cmd == "run" else cmd_demo(args)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
