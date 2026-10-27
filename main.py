"""Structured Extraction Service: resumes -> Resume JSON, model comparison.

    python main.py                      # full grid from config.yaml on all 60 texts + report
    python main.py run --model gpt-4o-mini --prompt v1 --sampling A [--split golden] [--limit 5] [--force]
    python main.py report               # results/summary.md from every run
    python main.py eval results/<run>   # recompute metrics of one run
    python main.py check                # validate golden set against the schema (no API calls)
"""
import argparse
import csv
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.extract import build_messages, extract, load_prompt, strict_schema
from src.llm import LLMClient
from src.metrics import FIELDS, summarize
from src.schema import Resume

ROOT = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8")


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def load_docs(cfg: dict, split: str) -> list[dict]:
    data = ROOT / cfg["data_dir"]
    rows = list(csv.DictReader((data / "manifest.csv").open(encoding="utf-8-sig")))
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    for r in rows:
        r["text"] = (data / "texts" / f"{r['doc_id']}.txt").read_text(encoding="utf-8")
    return rows


def server_up(base_url: str) -> bool:
    """Is a local OpenAI-compatible server (Ollama) answering at base_url?"""
    try:
        urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=3)
        return True
    except OSError:
        return False


def load_golden(cfg: dict) -> dict[str, dict]:
    folder = ROOT / cfg["data_dir"] / "golden"
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


def cmd_run(cfg, model, prompt, sampling, split="all", limit=None, force=False) -> dict:
    run = f"{model}__{prompt}__{sampling}"
    out_dir = ROOT / cfg["results_dir"] / run
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.jsonl"
    if force and out_file.exists():
        out_file.unlink()
    done = {r["doc_id"] for r in read_jsonl(out_file)}           # cache: rerun only what is missing

    client = LLMClient(model, cfg["models"][model])
    schema = strict_schema()
    guide = (ROOT / cfg["data_dir"] / "LABELING_GUIDE.md").read_text(encoding="utf-8")
    template = load_prompt(ROOT / "prompts", prompt)
    params = cfg["sampling"][sampling]
    docs = [d for d in load_docs(cfg, split) if d["doc_id"] not in done][:limit]

    def work(doc):
        messages = build_messages(template, doc["text"], doc["resume_date"], schema, guide)
        rec = extract(client, messages, schema, params)
        return {"doc_id": doc["doc_id"], "model": model, "model_id": client.model, "prompt_version": prompt,
                "sampling": sampling, "temperature": params["temperature"], "top_p": params["top_p"],
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **rec}

    workers = cfg["models"][model].get("workers", cfg.get("workers", 4))  # local model: 1, else latency includes queueing
    with ThreadPoolExecutor(workers) as pool, out_file.open("a", encoding="utf-8") as fh:
        for i, rec in enumerate(pool.map(work, docs), 1):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            status = "ok" if rec["valid_first"] else "repaired" if rec["repaired"] else "FAILED"
            print(f"  [{run}] {i}/{len(docs)} {rec['doc_id']} {status} {rec['latency_s']}s")
    return cmd_eval(cfg, out_dir)


def cmd_eval(cfg, out_dir: Path) -> dict:
    records = read_jsonl(out_dir / "results.jsonl")
    if not records:
        return {}
    metrics = summarize(records, load_golden(cfg))
    first = records[0]
    metrics["run"] = {k: first[k] for k in ("model", "model_id", "prompt_version", "sampling", "temperature", "top_p")}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{out_dir.name}: accuracy={metrics['field_accuracy']} valid={metrics['valid_pct']}% "
          f"failed={metrics['failed']} p50={metrics['latency_p50_s']}s $/1000={metrics['cost_per_1000_docs_usd']}")
    return metrics


def cmd_report(cfg) -> None:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / cfg["results_dir"]).glob("*/metrics.json"))]
    runs.sort(key=lambda m: m["field_accuracy"] or 0, reverse=True)
    lines = ["# Порівняння запусків", "",
             "Точність — середня по полях на golden set (20 резюме); решта метрик — на всіх документах запуску.", "",
             "| Модель | Промпт | Семплінг (t / top_p) | Точність | Валідні з 1-ї спроби | Валідні після repair | failed | p50, с | p95, с | $ / 1000 резюме |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in runs:
        r = m["run"]
        lines.append(f"| {r['model_id']} | {r['prompt_version']} | {r['sampling']} ({r['temperature']} / {r['top_p']}) | "
                     f"{m['field_accuracy']:.3f} | {m['valid_first_pct']}% | {m['valid_pct']}% | {m['failed']} | "
                     f"{m['latency_p50_s']} | {m['latency_p95_s']} | {m['cost_per_1000_docs_usd']} |")
    lines += ["", "## Точність по полях", "",
              "| Запуск | " + " | ".join(FIELDS) + " |", "|---|" + "---|" * len(FIELDS)]
    for m in runs:
        r = m["run"]
        lines.append(f"| {r['model']} {r['prompt_version']} {r['sampling']} | " +
                     " | ".join(f"{m['per_field'].get(f, 0):.2f}" for f in FIELDS) + " |")
    out = ROOT / cfg["results_dir"] / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {out.relative_to(ROOT)} ({len(runs)} runs)")


def cmd_check(cfg) -> None:
    golden = load_golden(cfg)
    for doc, data in golden.items():
        Resume.model_validate(data)
    docs = load_docs(cfg, "all")
    print(f"ok: {len(golden)} golden labels valid, {len(docs)} texts, "
          f"{sum(d['split'] == 'golden' for d in docs)} marked golden in manifest")


def cmd_grid(cfg) -> None:
    grid = cfg["grid"]
    for model in grid["models"]:
        mcfg = cfg["models"][model]
        if "api_key_env" in mcfg and not os.environ.get(mcfg["api_key_env"]):
            print(f"skip {model}: {mcfg['api_key_env']} is not set in .env")
            continue
        if "api_key_env" not in mcfg and not server_up(mcfg["base_url"]):
            print(f"skip {model}: local server {mcfg['base_url']} is not running (see README)")
            continue
        for prompt in mcfg.get("grid_prompts", grid["prompts"]):
            for sampling in grid["sampling"]:
                cmd_run(cfg, model, prompt, sampling, split=mcfg.get("grid_split", "all"))
    cmd_report(cfg)


def main() -> None:
    load_dotenv(ROOT / ".env")
    cfg = load_cfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    run = sub.add_parser("run")
    run.add_argument("--model", required=True, choices=list(cfg["models"]))
    run.add_argument("--prompt", default="v1")
    run.add_argument("--sampling", default="A", choices=list(cfg["sampling"]))
    run.add_argument("--split", default="all", choices=["all", "golden", "dev"])
    run.add_argument("--limit", type=int)
    run.add_argument("--force", action="store_true", help="drop cached results of this run")
    ev = sub.add_parser("eval")
    ev.add_argument("run_dir")
    sub.add_parser("report")
    sub.add_parser("check")
    sub.add_parser("grid")
    args = ap.parse_args()

    if args.cmd == "run":
        cmd_run(cfg, args.model, args.prompt, args.sampling, args.split, args.limit, args.force)
    elif args.cmd == "eval":
        cmd_eval(cfg, Path(args.run_dir))
    elif args.cmd == "report":
        cmd_report(cfg)
    elif args.cmd == "check":
        cmd_check(cfg)
    else:
        cmd_grid(cfg)


if __name__ == "__main__":
    main()
