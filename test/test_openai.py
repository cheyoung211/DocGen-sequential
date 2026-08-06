"""Smoke test for the project's GPT-4o-mini LLM client."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.llm_client import LLMClient


def test_gpt4o_mini() -> None:
    client = LLMClient()
    response = client.generate(
        system_prompt="You are a helpful assistant for research on multi-agent systems.",
        user_prompt=(
            "Briefly describe the role of a planner agent in a LaTeX document "
            "generation system. Answer in English, in three sentences."
        ),
        temperature=0.2,
        max_new_tokens=128,
    )
    assert response


if __name__ == "__main__":
    test_gpt4o_mini()
    print("GPT-4o mini API smoke test passed.")
