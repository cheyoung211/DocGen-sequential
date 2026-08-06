# document_generation/run_text_agent_demo.py

from __future__ import annotations

import json
from pathlib import Path

from schemas import PlannerOutput, TextContext
from agents.text_agent import TextAgent


def main():
    # 1) plan.json 읽기
    req_id = "req_20260208_001"
    plan_path = Path(f"outputs/{req_id}/plan.json")
    plan = PlannerOutput.parse_file(plan_path)

    # 2) TextAgent 준비
    agent = TextAgent()

    # 3) context (retrieval은 아직 구현 안 했으니 빈 값)
    context = TextContext(
        user_query="Write an internal report comparing normal/under/over preprocessing for IR.",
        retrieved_snippets=[],
    )

    # 4) 각 섹션에 대해 텍스트 생성 + tex 파일로 저장
    sections_dir = Path(f"outputs/{req_id}/sections")
    sections_dir.mkdir(parents=True, exist_ok=True)

    for sec in plan.doc_outline:
        print(f"\n=== Generating section: {sec.section_id} ===")
        out = agent.generate_section(
            request_id=req_id,
            section_spec=sec,
            context=context,
            out_dir=sections_dir,
        )
        print(f"Saved {sec.section_id}.tex")

    print("\nAll sections generated!")


if __name__ == "__main__":
    main()
