"""Field accuracy against the golden set + validity, latency and cost of a run.

Scoring per field (0..1), averaged over golden documents:
- scalars (title, age, city, desired_salary, driver_license): exact match after normalisation, null == null;
- string/enum lists (employment_type, work_location_preference, additional_educations, skills): set F1;
- languages: set F1 over (language, level) pairs;
- work_experiences / educations: entries matched greedily, accuracy over sub-fields,
  divided by max(len(gold), len(pred)) so missing and invented entries both cost;
- empty gold and empty prediction count as 1.0.
responsibilities and missing_fields are not scored (free text / derived).
"""
import re
import statistics

SCALARS = ["title", "age", "city", "desired_salary", "driver_license"]
SETS = ["employment_type", "work_location_preference", "additional_educations", "skills"]
EXP_KEYS = ["position", "company", "city", "start_date", "end_date", "is_current"]
EDU_KEYS = ["institution", "faculty", "city", "level", "start_year", "end_year"]
FIELDS = SCALARS + SETS + ["languages", "work_experiences", "educations"]


def norm(value):
    if isinstance(value, str):
        value = re.sub(r"[\s ﻿]+", " ", value).strip(" .,;:").lower()
        return value.replace("’", "'").replace("ʼ", "'")
    return value


def set_f1(gold, pred) -> float:
    g, p = {norm(x) for x in gold}, {norm(x) for x in pred}
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    tp = len(g & p)
    return 0.0 if tp == 0 else 2 * tp / (len(g) + len(p))


def list_score(gold: list, pred: list, keys: list) -> float:
    if not gold and not pred:
        return 1.0
    remaining, hits = list(pred), 0
    for g in gold:
        best, best_hits = None, -1
        for p in remaining:
            h = sum(norm(g.get(k)) == norm(p.get(k)) for k in keys)
            if h > best_hits:
                best, best_hits = p, h
        if best is not None:
            remaining.remove(best)
            hits += best_hits
    return hits / (max(len(gold), len(pred)) * len(keys))


def score_doc(gold: dict, pred: dict) -> dict:
    s = {k: float(norm(gold[k]) == norm(pred.get(k))) for k in SCALARS}
    s.update({k: set_f1(gold[k], pred.get(k) or []) for k in SETS})
    pairs = lambda rows: [f"{r['language']}|{r['level']}" for r in rows or []]  # noqa: E731
    s["languages"] = set_f1(pairs(gold["languages"]), pairs(pred.get("languages")))
    s["work_experiences"] = list_score(gold["work_experiences"], pred.get("work_experiences") or [], EXP_KEYS)
    s["educations"] = list_score(gold["educations"], pred.get("educations") or [], EDU_KEYS)
    return s


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summarize(records: list[dict], golden: dict[str, dict]) -> dict:
    """records: one per document of a run. Failed outputs score 0 on every field."""
    per_field = {f: [] for f in FIELDS}
    per_doc = {}
    for r in records:
        if r["doc_id"] not in golden:
            continue
        s = score_doc(golden[r["doc_id"]], r["output"]) if r["output"] else {f: 0.0 for f in FIELDS}
        per_doc[r["doc_id"]] = round(statistics.mean(s.values()), 4)
        for f, v in s.items():
            per_field[f].append(v)
    n = len(records)
    lat = [r["latency_s"] for r in records if r["calls"]]
    cost = sum(r["cost_usd"] for r in records)
    return {
        "docs": n,
        "golden_docs": len(per_doc),
        "field_accuracy": round(statistics.mean(per_doc.values()), 4) if per_doc else None,
        "per_field": {f: round(statistics.mean(v), 4) for f, v in per_field.items() if v},
        "per_doc": per_doc,
        "valid_first_pct": round(100 * sum(r["valid_first"] for r in records) / n, 1),
        "valid_pct": round(100 * sum(r["valid"] for r in records) / n, 1),
        "repaired": sum(r["repaired"] for r in records),
        "failed": sum(r["failed"] for r in records),
        "retries": sum(r["retries"] for r in records),
        "latency_p50_s": round(percentile(lat, 0.5), 2),
        "latency_p95_s": round(percentile(lat, 0.95), 2),
        "cost_usd": round(cost, 4),
        "cost_per_1000_docs_usd": round(1000 * cost / n, 3) if n else None,
        "tokens_in": sum(r["input_tokens"] for r in records),
        "tokens_out": sum(r["output_tokens"] for r in records),
    }
