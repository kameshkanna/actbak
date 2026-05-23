"""Download and pre-cache all datasets required by the activation-baking pipeline.

Dataset inventory
-----------------
  mtbench          MT-Bench 80-question set  →  data/mtbench_questions.jsonl
                   Used by: experiment 01 (norm profiling)

  harmbench_eval   HarmBench standard behaviors (from GitHub CSV)
                   →  data/eval_prompts/harmbench.jsonl
                   Used by: experiment 03 (safety eval), ablation 01

  clearharm_eval   ClearHarm (AlignmentResearch/ClearHarm, HuggingFace)
                   →  data/eval_prompts/clearharm.jsonl
                   Used by: experiment 03 (safety eval)

  sycophancy_eval  Anthropic model-written sycophancy evals (HuggingFace)
                   →  data/eval_prompts/sycophancy.jsonl
                   Used by: experiment 03 (sycophancy eval)

  gsm8k            GSM8K test set (HuggingFace cache)
  mmlu             MMLU all/test (HuggingFace cache)
  truthfulqa       TruthfulQA multiple_choice/validation (HuggingFace cache)

Usage:
    python data/download_datasets.py

    python data/download_datasets.py --n-harmbench 100 --n-clearharm 100 --n-sycophancy 100

    python data/download_datasets.py --only mtbench harmbench_eval clearharm_eval sycophancy_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent
EVAL_PROMPTS_DIR = DATA_DIR / "eval_prompts"

MTBENCH_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)
HARMBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)

# Anthropic model-written-evals — NLP survey sycophancy (most diverse subset)
ANTHROPIC_SYCO_FILES = [
    "sycophancy/sycophancy_on_nlp_survey.jsonl",
    "sycophancy/sycophancy_on_philpapers_survey.jsonl",
    "sycophancy/sycophancy_on_political_typology_quiz.jsonl",
]
ANTHROPIC_SYCO_HF_REPO = "Anthropic/model-written-evals"

ALL_DATASETS = [
    "mtbench",
    "harmbench_eval",
    "clearharm_eval",
    "sycophancy_eval",
    "gsm8k",
    "mmlu",
    "truthfulqa",
]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_counts(config_path: Path) -> dict[str, int]:
    """Read sample counts from experiment.yml, falling back to hard-coded defaults."""
    defaults: dict[str, int] = {
        "n_harmbench": 200,
        "n_clearharm": 200,
        "n_sycophancy": 200,
        "seed": 42,
    }
    if not config_path.exists():
        logger.warning("Config not found at %s — using defaults.", config_path)
        return defaults

    with config_path.open() as f:
        raw = yaml.safe_load(f)
    exp = raw.get("experiment", {})
    return {
        "n_harmbench":  exp.get("n_harmbench",  defaults["n_harmbench"]),
        "n_clearharm":  exp.get("n_clearharm",  defaults["n_clearharm"]),
        "n_sycophancy": exp.get("n_sycophancy", defaults["n_sycophancy"]),
        "seed":         exp.get("seed",         defaults["seed"]),
    }


# ---------------------------------------------------------------------------
# MT-Bench
# ---------------------------------------------------------------------------


def download_mtbench(output: Path = DATA_DIR / "mtbench_questions.jsonl") -> None:
    """Fetch MT-Bench 80 first-turn questions from the FastChat repository."""
    if output.exists():
        logger.info("MT-Bench already at %s — skipping.", output)
        return

    logger.info("Downloading MT-Bench questions...")
    resp = requests.get(MTBENCH_URL, timeout=30)
    resp.raise_for_status()

    questions: list[dict] = []
    for line in resp.text.strip().splitlines():
        obj = json.loads(line)
        questions.append({
            "question_id": obj["question_id"],
            "category":    obj["category"],
            "prompt":      obj["turns"][0],
        })

    output.write_text(
        "\n".join(json.dumps(q) for q in questions), encoding="utf-8"
    )
    logger.info("MT-Bench: %d questions → %s", len(questions), output)


# ---------------------------------------------------------------------------
# HarmBench
# ---------------------------------------------------------------------------


def download_harmbench_eval(n: int, seed: int) -> list[dict]:
    """Fetch HarmBench standard behaviors CSV from GitHub and sample n rows.

    Args:
        n:    Number of prompts to sample.
        seed: Random seed for reproducible sampling.

    Returns:
        List of dicts with keys ``source``, ``id``, ``prompt``, ``category``.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    logger.info("Downloading HarmBench behaviors CSV...")
    resp = requests.get(HARMBENCH_CSV_URL, timeout=30)
    resp.raise_for_status()

    rows: list[dict] = []
    lines = resp.text.strip().splitlines()
    header = [h.strip().strip('"') for h in lines[0].split(",")]

    try:
        bid_idx      = header.index("BehaviorID")
        behavior_idx = header.index("Behavior")
        func_idx     = header.index("FunctionalCategory")
        sem_idx      = header.index("SemanticCategory")
    except ValueError as exc:
        raise RuntimeError(
            f"Unexpected HarmBench CSV header: {header}. Missing column: {exc}"
        ) from exc

    for line in lines[1:]:
        parts = line.split(",", len(header) - 1)
        if len(parts) < len(header):
            continue
        if parts[func_idx].strip().strip('"') != "standard":
            continue
        rows.append({
            "source":   "harmbench",
            "id":       parts[bid_idx].strip().strip('"'),
            "prompt":   parts[behavior_idx].strip().strip('"'),
            "category": parts[sem_idx].strip().strip('"'),
        })

    rng = random.Random(seed)
    sampled = rng.sample(rows, min(n, len(rows)))
    logger.info("HarmBench: %d standard behaviors → sampled %d", len(rows), len(sampled))
    return sampled


# ---------------------------------------------------------------------------
# ClearHarm
# ---------------------------------------------------------------------------


def download_clearharm_eval(n: int, seed: int) -> list[dict]:
    """Load ClearHarm from HuggingFace and sample n records.

    Args:
        n:    Number of prompts to sample.
        seed: Random seed for reproducible sampling.

    Returns:
        List of dicts with keys ``source``, ``id``, ``prompt``.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    logger.info("Loading ClearHarm (AlignmentResearch/ClearHarm)...")
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
        ds = load_dataset("AlignmentResearch/ClearHarm", split="train")
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Failed to load ClearHarm: {exc}") from exc

    rows: list[dict] = []
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


# ---------------------------------------------------------------------------
# Sycophancy (Anthropic model-written-evals)
# ---------------------------------------------------------------------------


def download_sycophancy_eval(n: int, seed: int) -> list[dict]:
    """Load Anthropic model-written sycophancy evals from HuggingFace and sample n records.

    Pulls from three sycophancy sub-files (NLP survey, PhilPapers survey,
    political typology quiz), merges them, then samples n uniformly.

    Each output record contains:
        source               always ``"anthropic_syco"``
        id                   unique identifier
        prompt               the user question (includes the user's stated view)
        sycophantic_answer   what a sycophantic model would say
        honest_answer        what an honest model would say

    The ``sycophantic_answer`` / ``honest_answer`` fields are passed to the
    judge so it can make well-anchored binary decisions without free-form rubric
    interpretation.

    Args:
        n:    Number of prompts to sample.
        seed: Random seed for reproducible sampling.

    Returns:
        List of dicts with keys described above.

    Raises:
        RuntimeError: If the HuggingFace dataset cannot be loaded.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    logger.info("Loading Anthropic model-written sycophancy evals from HuggingFace...")
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("datasets package not installed. Run: pip install datasets") from exc

    rows: list[dict] = []
    for file_path in ANTHROPIC_SYCO_FILES:
        try:
            ds = load_dataset(
                ANTHROPIC_SYCO_HF_REPO,
                data_files={"train": file_path},
                split="train",
            )
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("Could not load %s: %s — skipping.", file_path, exc)
            continue

        stem = Path(file_path).stem
        for i, ex in enumerate(ds):
            question = ex.get("question", "").strip()
            if not question:
                continue
            rows.append({
                "source":             "anthropic_syco",
                "id":                 f"{stem}_{i}",
                "prompt":             question,
                "sycophantic_answer": ex.get("answer_matching_behavior", "").strip(),
                "honest_answer":      ex.get("answer_not_matching_behavior", "").strip(),
            })

    if not rows:
        raise RuntimeError(
            "No sycophancy records loaded. "
            f"Check that '{ANTHROPIC_SYCO_HF_REPO}' is accessible on HuggingFace."
        )

    rng = random.Random(seed)
    sampled = rng.sample(rows, min(n, len(rows)))
    logger.info(
        "Sycophancy eval: %d records across %d files → sampled %d",
        len(rows), len(ANTHROPIC_SYCO_FILES), len(sampled),
    )
    return sampled


# ---------------------------------------------------------------------------
# HuggingFace dataset pre-caching
# ---------------------------------------------------------------------------


def _cache_hf_dataset(path: str, name: str | None, split: str, label: str) -> None:
    """Load a HuggingFace dataset once to populate the local cache."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
        logger.info("Pre-caching %s (%s / %s)...", label, path, split)
        ds = load_dataset(path, name, split=split)
        logger.info("%s: %d examples cached.", label, len(ds))
    except Exception as exc:
        logger.warning("Could not cache %s: %s — will download at eval time.", label, exc)


def cache_gsm8k() -> None:
    _cache_hf_dataset("openai/gsm8k", "main", "test", "GSM8K")


def cache_mmlu() -> None:
    _cache_hf_dataset("cais/mmlu", "all", "test", "MMLU")


def cache_truthfulqa() -> None:
    _cache_hf_dataset("truthfulqa/truthful_qa", "multiple_choice", "validation", "TruthfulQA")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved %d records → %s", len(records), path)


def _already_exists(path: Path, label: str) -> bool:
    if path.exists():
        logger.info("%s already at %s — skipping.", label, path)
        return True
    return False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and pre-cache all datasets for activation-baking experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config/experiment.yml",
        help="Path to experiment.yml; used to read default sample counts.",
    )
    parser.add_argument("--n-harmbench",  type=int, default=None)
    parser.add_argument("--n-clearharm",  type=int, default=None)
    parser.add_argument("--n-sycophancy", type=int, default=None)
    parser.add_argument("--seed",         type=int, default=None)
    parser.add_argument(
        "--only", nargs="+", choices=ALL_DATASETS, metavar="DATASET",
        help=f"Download only specified dataset(s). Choices: {ALL_DATASETS}. Default: all.",
    )
    parser.add_argument(
        "--skip-hf-cache", action="store_true",
        help="Skip pre-caching gsm8k/mmlu/truthfulqa. They download lazily on first eval.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the output file already exists.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    counts = load_counts(Path(args.config))
    n_harmbench  = args.n_harmbench  if args.n_harmbench  is not None else counts["n_harmbench"]
    n_clearharm  = args.n_clearharm  if args.n_clearharm  is not None else counts["n_clearharm"]
    n_sycophancy = args.n_sycophancy if args.n_sycophancy is not None else counts["n_sycophancy"]
    seed         = args.seed         if args.seed         is not None else counts["seed"]

    targets: set[str] = set(args.only) if args.only else set(ALL_DATASETS)
    if args.skip_hf_cache:
        targets -= {"gsm8k", "mmlu", "truthfulqa"}

    logger.info(
        "Datasets: %s  |  n_harmbench=%d  n_clearharm=%d  n_sycophancy=%d  seed=%d",
        sorted(targets), n_harmbench, n_clearharm, n_sycophancy, seed,
    )

    if "mtbench" in targets:
        out = DATA_DIR / "mtbench_questions.jsonl"
        if args.force and out.exists():
            out.unlink()
        download_mtbench(out)

    if "harmbench_eval" in targets:
        out = EVAL_PROMPTS_DIR / "harmbench.jsonl"
        if args.force and out.exists():
            out.unlink()
        if not _already_exists(out, "HarmBench eval prompts"):
            _save_jsonl(download_harmbench_eval(n_harmbench, seed), out)

    if "clearharm_eval" in targets:
        out = EVAL_PROMPTS_DIR / "clearharm.jsonl"
        if args.force and out.exists():
            out.unlink()
        if not _already_exists(out, "ClearHarm eval prompts"):
            _save_jsonl(download_clearharm_eval(n_clearharm, seed), out)

    if "sycophancy_eval" in targets:
        out = EVAL_PROMPTS_DIR / "sycophancy.jsonl"
        if args.force and out.exists():
            out.unlink()
        if not _already_exists(out, "Sycophancy eval prompts"):
            _save_jsonl(download_sycophancy_eval(n_sycophancy, seed), out)

    if "gsm8k" in targets:
        cache_gsm8k()
    if "mmlu" in targets:
        cache_mmlu()
    if "truthfulqa" in targets:
        cache_truthfulqa()

    logger.info("All done. Data directory: %s", DATA_DIR.resolve())


if __name__ == "__main__":
    main()
