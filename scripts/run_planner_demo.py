# document_generation/run_planner_demo.py

from __future__ import annotations

from pathlib import Path

from schemas import (
    PlannerInput,
    TemplateSpec,
    FileInput,
)
from agents.planner import PlannerAgent


def main() -> None:
    # 1) PlannerInput 예제 구성
    planner_input = PlannerInput(
        request_id="req_20260208_001",
        language="en",
        doc_type="internal_report",
        user_query=(
            "Write an internal report comparing normal, underprocessed, and "
            "overprocessed preprocessing settings for our IR experiments. "
            "Highlight differences in ranking metrics such as nDCG and pairwise F1."
        ),
        inputs=[
            FileInput(
                type="file",
                name="IR metrics CSV",
                path="data/metrics.csv",
            )
        ],
        template=TemplateSpec(
            name="internal_report_v1",
            main_tex_template="templates/internal_report_v1/main.tex",
            section_order=["intro", "method", "results", "conclusion"],
        ),
    )

    # 2) PlannerAgent 실행
    agent = PlannerAgent()
    plan = agent.plan(planner_input)

    # 3) 콘솔에 간단히 출력
    print("=== Planner output (sections) ===")
    for sec in plan.doc_outline:
        print(
            f"- {sec.section_id}: {sec.title} "
            f"(figures={sec.needs_figures}, retrieval={sec.needs_retrieval})"
        )

    # 4) plan.json 저장
    out_dir = Path("outputs") / planner_input.request_id
    plan_path = agent.save_plan(plan, out_dir)
    print(f"\nSaved plan to: {plan_path}")


if __name__ == "__main__":
    main()
