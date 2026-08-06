# # run_latex_integrator_demo.py

# from __future__ import annotations

# from pathlib import Path

# from schemas import PlannerOutput
# from agents.latex_integrator import LatexIntegrator


# def main():
#     # 1) request ID와 프로젝트 디렉토리 설정
#     req_id = "req_20260208_001"  # 너가 planner/text/table 데모에 사용한 것과 동일하게
#     project_dir = Path("outputs") / req_id

#     # 2) plan.json 로드
#     plan_path = project_dir / "plan.json"
#     plan = PlannerOutput.parse_file(plan_path)

#     # 3) LatexIntegrator 생성
#     integrator = LatexIntegrator(engine="pdflatex", runs=2)

#     # 4) summary 구성
#     summary = integrator.build_summary(plan, project_dir)

#     # 5) main.tex 생성
#     main_tex_path = integrator.write_main_tex(
#         summary,
#         title="IR Preprocessing Comparison Report",
#     )
#     print(f"main.tex written to: {main_tex_path}")

#     # 6) 컴파일 시도 (LaTeX 엔진이 없으면 그냥 log에만 기록)
#     summary = integrator.compile(summary)

#     print(f"compile_attempted: {summary.compile_attempted}")
#     print(f"compile_success:   {summary.compile_success}")
#     print(f"compile_log_path:  {summary.compile_log_path}")

#     if summary.compile_success:
#         pdf_path = project_dir / "main.pdf"
#         print(f"PDF should be at: {pdf_path}")
#     else:
#         print("Compilation failed or LaTeX engine not found. Check compile.log.")


# if __name__ == "__main__":
#     main()


# run_latex_integrator_demo.py

from __future__ import annotations

from pathlib import Path

from schemas import PlannerOutput
from agents.latex_integrator import LatexIntegrator


def main():
    req_id = "req_20260208_001"  # 너가 사용하는 request_id와 맞게
    project_dir = Path("outputs") / req_id

    plan_path = project_dir / "plan.json"
    plan = PlannerOutput.parse_file(plan_path)

    integrator = LatexIntegrator(engine="pdflatex", runs=2)

    summary = integrator.build_summary(plan, project_dir)

    # 템플릿 경로 (있으면 쓰고, 없으면 자동 fallback)
    template_path = Path("templates/default/main.tex")
    if template_path.is_file():
        print(f"Using template: {template_path}")
        main_tex_path = integrator.write_main_tex(
            summary,
            title="IR Preprocessing Comparison Report",
            template_path=template_path,
        )
    else:
        print("No template found, using default article template.")
        main_tex_path = integrator.write_main_tex(
            summary,
            title="IR Preprocessing Comparison Report",
            template_path=None,  # 또는 그냥 생략해도 됨
        )

    print(f"main.tex written to: {main_tex_path}")

    summary = integrator.compile(summary)

    print(f"compile_attempted: {summary.compile_attempted}")
    print(f"compile_success:   {summary.compile_success}")
    print(f"compile_log_path:  {summary.compile_log_path}")

    if summary.compile_success:
        pdf_path = project_dir / "main.pdf"
        print(f"PDF should be at: {pdf_path}")
    else:
        print("Compilation failed or LaTeX engine not found. Check compile.log.")


if __name__ == "__main__":
    main()
