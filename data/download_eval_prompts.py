"""Download and sample evaluation prompts from HarmBench and ClearHarm.

HarmBench  — fetched from the centerforaisafety/HarmBench GitHub repo.
ClearHarm  — loaded from AlignmentResearch/ClearHarm on HuggingFace.

Outputs (10 samples each, reproducible seed):
    data/eval_prompts/harmbench.jsonl
    data/eval_prompts/clearharm.jsonl

Usage:
    python data/download_eval_prompts.py [--n 10] [--seed 42]
"""

import argparse
import json
import logging
import random
from pathlib import Path

import requests
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

HARMBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
OUT_DIR = Path("data/eval_prompts")


def download_harmbench(n: int, seed: int) -> list[dict]:
    """Fetch HarmBench behaviors CSV, filter standard (no context), sample n."""
    logger.info("Downloading HarmBench from GitHub...")
    resp = requests.get(HARMBENCH_CSV_URL, timeout=30)
    resp.raise_for_status()

    rows = []
    lines = resp.text.strip().splitlines()
    header = [h.strip() for h in lines[0].split(",")]
    behavior_idx = header.index("Behavior")
    func_idx = header.index("FunctionalCategory")
    sem_idx = header.index("SemanticCategory")
    bid_idx = header.index("BehaviorID")

    for line in lines[1:]:
        # CSV may have commas inside quoted fields — use a simple split on first fields
        parts = line.split(",", len(header) - 1)
        if len(parts) < len(header):
            continue
        func_cat = parts[func_idx].strip().strip('"')
        if func_cat != "standard":
            continue
        rows.append({
            "source":   "harmbench",
            "id":       parts[bid_idx].strip().strip('"'),
            "prompt":   parts[behavior_idx].strip().strip('"'),
            "category": parts[sem_idx].strip().strip('"'),
        })

    rng = random.Random(seed)
    sampled = rng.sample(rows, min(n, len(rows)))
    logger.info("HarmBench: %d standard rows → sampled %d", len(rows), len(sampled))
    return sampled


def download_clearharm(n: int, seed: int) -> list[dict]:
    """Load ClearHarm from HuggingFace, sample n from default split."""
    logger.info("Loading ClearHarm from HuggingFace (AlignmentResearch/ClearHarm)...")
    ds = load_dataset("AlignmentResearch/ClearHarm", split="train")

    rows = []
    for i, ex in enumerate(ds):
        content = ex.get("content", [])
        prompt = content[0] if isinstance(content, list) and content else str(content)
        prompt = prompt.strip()
        if not prompt:
            continue
        rows.append({
            "source": "clearharm",
            "id":     f"clearharm_{i}",
            "prompt": prompt,
        })

    rng = random.Random(seed)
    sampled = rng.sample(rows, min(n, len(rows)))
    logger.info("ClearHarm: %d rows → sampled %d", len(rows), len(sampled))
    return sampled


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved %d records → %s", len(records), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HarmBench + ClearHarm eval prompts")
    parser.add_argument("--n", type=int, default=10, help="Samples per benchmark")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    harmbench = download_harmbench(args.n, args.seed)
    save_jsonl(harmbench, OUT_DIR / "harmbench.jsonl")

    clearharm = download_clearharm(args.n, args.seed)
    save_jsonl(clearharm, OUT_DIR / "clearharm.jsonl")

    logger.info("Done. Run experiments/03_ramp_steering_eval.py next.")


if __name__ == "__main__":
    main()
