"""
Analysis script for hesm_conv-26.json
- Overall statistics and per-category breakdown
- Metrics excluding Unknown predictions
- Evidence recall: how many questions have evidence in retrieved_ids
"""

import json
from collections import defaultdict

DATA_PATH = "d:/code/hesm/experiments/outputs/locomo/answers/hesm_conv-26.json"

CATEGORY_NAMES = {
    1: "Single-hop",
    2: "Multi-hop",
    3: "Open-ended",
    4: "Temporal",
    5: "Adversarial",
}


def avg(values):
    return sum(values) / len(values) if values else 0.0


def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def analyze_retrieval_metrics(records, label=""):
    """Average retrieval_metrics across records for @1, @3, @5."""
    result = {}
    for k in ("1", "3", "5"):
        recalls, precisions, f1s, accuracies = [], [], [], []
        for r in records:
            rm = r.get("retrieval_metrics", {}).get(k, {})
            if rm:
                recalls.append(rm.get("recall", 0))
                precisions.append(rm.get("precision", 0))
                f1s.append(rm.get("f1", 0))
                accuracies.append(rm.get("accuracy", 0))
        result[k] = {
            "recall":    avg(recalls),
            "precision": avg(precisions),
            "f1":        avg(f1s),
            "accuracy":  avg(accuracies),
        }
    return result


def print_retrieval_metrics(rm):
    for k in ("1", "3", "5"):
        m = rm[k]
        print(f"    @{k:>2}: recall={m['recall']:.4f}  precision={m['precision']:.4f}"
              f"  f1={m['f1']:.4f}  accuracy={m['accuracy']:.4f}")


def evidence_in_retrieved(record):
    """Return (any_hit, all_hit) booleans for whether evidence docs are in retrieved_ids."""
    evidence = record.get("evidence") or []
    retrieved = set(record.get("retrieved_ids") or [])
    if not evidence:
        return False, False
    hits = sum(1 for e in evidence if e in retrieved)
    return hits > 0, hits == len(evidence)


def main():
    data = load_data()
    meta = {k: v for k, v in data.items() if k != "records"}
    records = list(data["records"].values())

    total = len(records)
    unknown_recs  = [r for r in records if r.get("prediction") == "Unknown"]
    answered_recs = [r for r in records if r.get("prediction") != "Unknown"]

    print_section("FILE METADATA")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------ #
    # 1. Overall counts
    # ------------------------------------------------------------------ #
    print_section("OVERALL COUNTS")
    print(f"  Total questions      : {total}")
    print(f"  Unknown predictions  : {len(unknown_recs):>4}  ({100*len(unknown_recs)/total:.1f}%)")
    print(f"  Answered predictions : {len(answered_recs):>4}  ({100*len(answered_recs)/total:.1f}%)")

    by_cat_all      = defaultdict(list)
    by_cat_answered = defaultdict(list)
    for r in records:
        by_cat_all[r.get("category")].append(r)
    for r in answered_recs:
        by_cat_answered[r.get("category")].append(r)

    print(f"\n  {'Category':<28} {'Total':>6} {'Unknown':>8} {'Answered':>9}")
    print(f"  {'-'*55}")
    for cat in sorted(by_cat_all):
        name = CATEGORY_NAMES.get(cat, f"Cat{cat}")
        tot  = len(by_cat_all[cat])
        unk  = tot - len(by_cat_answered[cat])
        ans  = len(by_cat_answered[cat])
        print(f"  {name:<28} {tot:>6} {unk:>8} {ans:>9}")

    # ------------------------------------------------------------------ #
    # 2. Quality metrics — answered only
    # ------------------------------------------------------------------ #
    print_section("QUALITY METRICS  (Unknown excluded)")

    def quality_block(recs, indent="  "):
        if not recs:
            print(f"{indent}(no records)")
            return
        f1s   = [r["f1"] for r in recs]
        precs = [r["f1_precision"] for r in recs]
        recs_ = [r["f1_recall"] for r in recs]
        judges = [r["judge_score"] for r in recs]
        print(f"{indent}N                : {len(recs)}")
        print(f"{indent}F1 (avg)         : {avg(f1s):.4f}")
        print(f"{indent}F1 Precision     : {avg(precs):.4f}")
        print(f"{indent}F1 Recall        : {avg(recs_):.4f}")
        print(f"{indent}Judge Score (avg): {avg(judges):.4f}")
        rm = analyze_retrieval_metrics(recs)
        print(f"{indent}Retrieval metrics:")
        print_retrieval_metrics(rm)

    print("\n  --- All answered ---")
    quality_block(answered_recs)

    for cat in sorted(by_cat_answered):
        name = CATEGORY_NAMES.get(cat, f"Cat{cat}")
        recs_cat = by_cat_answered[cat]
        print(f"\n  --- {name} (cat {cat}) ---")
        quality_block(recs_cat)

    # ------------------------------------------------------------------ #
    # 3. Quality metrics — all (including Unknown, for reference)
    # ------------------------------------------------------------------ #
    print_section("QUALITY METRICS  (all records, including Unknown)")
    quality_block(records)

    # ------------------------------------------------------------------ #
    # 4. Evidence recall: evidence appearing in retrieved_ids
    # ------------------------------------------------------------------ #
    print_section("EVIDENCE IN RETRIEVED_IDS")

    has_evidence = [r for r in records if r.get("evidence")]
    no_evidence  = [r for r in records if not r.get("evidence")]

    any_hits = 0
    all_hits = 0
    for r in has_evidence:
        any_hit, all_hit = evidence_in_retrieved(r)
        any_hits += any_hit
        all_hits += all_hit

    n_evid = len(has_evidence)
    print(f"  Questions with evidence field    : {n_evid} / {total}")
    print(f"  Questions with NO evidence field : {len(no_evidence)} / {total}")
    print()
    print(f"  At least 1 evidence in retrieved_ids : {any_hits:>4} / {n_evid}"
          f"  ({100*any_hits/n_evid:.1f}%)" if n_evid else "")
    print(f"  ALL evidence in retrieved_ids        : {all_hits:>4} / {n_evid}"
          f"  ({100*all_hits/n_evid:.1f}%)" if n_evid else "")

    # Per-category breakdown
    print(f"\n  {'Category':<28} {'Has-evid':>9} {'Any-hit':>8} {'Any%':>6} {'All-hit':>8} {'All%':>6}")
    print(f"  {'-'*68}")
    by_cat_evid = defaultdict(list)
    for r in has_evidence:
        by_cat_evid[r.get("category")].append(r)
    for cat in sorted(by_cat_evid):
        name = CATEGORY_NAMES.get(cat, f"Cat{cat}")
        recs_cat = by_cat_evid[cat]
        n = len(recs_cat)
        any_h = sum(1 for r in recs_cat if evidence_in_retrieved(r)[0])
        all_h = sum(1 for r in recs_cat if evidence_in_retrieved(r)[1])
        print(f"  {name:<28} {n:>9} {any_h:>8} {100*any_h/n:>5.1f}%"
              f" {all_h:>8} {100*all_h/n:>5.1f}%")

    # ------------------------------------------------------------------ #
    # 5. Misc stats
    # ------------------------------------------------------------------ #
    print_section("MISC STATS  (all records)")
    comp_ratios = [r.get("compression_ratio", 0) for r in records]
    latencies   = [r.get("latency_ms", 0) for r in records]
    ret_tokens  = [r.get("retrieved_tokens", 0) for r in records]
    print(f"  Compression ratio  avg={avg(comp_ratios):.2f}  "
          f"min={min(comp_ratios):.2f}  max={max(comp_ratios):.2f}")
    print(f"  Latency (ms)       avg={avg(latencies):.0f}  "
          f"min={min(latencies):.0f}  max={max(latencies):.0f}")
    print(f"  Retrieved tokens   avg={avg(ret_tokens):.1f}  "
          f"min={min(ret_tokens)}  max={max(ret_tokens)}")


if __name__ == "__main__":
    main()
