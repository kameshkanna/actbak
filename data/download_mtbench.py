"""Download MT-Bench questions from the FastChat repository.

Saves all 80 first-turn questions to data/mtbench_questions.jsonl.
Run once before experiments: python data/download_mtbench.py
"""

import json
import logging
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MTBENCH_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)
OUTPUT_PATH = Path(__file__).parent / "mtbench_questions.jsonl"


def download_mtbench(url: str = MTBENCH_URL, output: Path = OUTPUT_PATH) -> int:
    """Fetch MT-Bench questions and persist to disk.

    Args:
        url: Raw URL of the MT-Bench question.jsonl file.
        output: Destination path.

    Returns:
        Number of questions saved.
    """
    logger.info("Fetching MT-Bench questions from FastChat...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    questions: list[dict] = []
    for line in response.text.strip().splitlines():
        obj = json.loads(line)
        questions.append({
            "question_id": obj["question_id"],
            "category": obj["category"],
            "prompt": obj["turns"][0],
        })

    output.write_text(
        "\n".join(json.dumps(q) for q in questions),
        encoding="utf-8",
    )
    logger.info("Saved %d questions to %s", len(questions), output)
    return len(questions)


if __name__ == "__main__":
    count = download_mtbench()
    sys.exit(0 if count > 0 else 1)
