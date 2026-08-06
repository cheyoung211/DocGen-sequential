import os
import subprocess
import pandas as pd
import inspect
import functools
from typing import Any, Dict, List, Optional, Callable
from pydantic import validate_arguments, BaseModel, Field

# --- Tool Decorator and Registry ---

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}

def tool(func: Callable):
    """
    Decorator that registers a function as an agent tool.
    It extracts the docstring and type hints to create a schema for the LLM.
    """
    @functools.wraps(func)
    @validate_arguments
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return {"success": True, "output": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Extract metadata for LLM Function Calling
    sig = inspect.signature(func)
    TOOL_REGISTRY[func.__name__] = {
        "name": func.__name__,
        "description": inspect.getdoc(func) or "No description provided.",
        "parameters": {
            "type": "object",
            "properties": {
                name: {"type": str(param.annotation.__name__)} # Simplified type mapping
                for name, param in sig.parameters.items()
            },
            "required": [
                name for name, param in sig.parameters.items() 
                if param.default == inspect.Parameter.empty
            ]
        }
    }
    return wrapper

# --- Core Business Tools ---

@tool
def pandas_csv_to_latex_table(file_path: str, caption: str, label: str) -> str:
    """
    Reads a CSV file and converts it into a high-quality LaTeX 'booktabs' table.
    Useful for business reports requiring precise data presentation.
    
    Args:
        file_path: Path to the input CSV file.
        caption: The caption for the LaTeX table.
        label: The LaTeX label for cross-referencing (e.g., tab:market_share).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at: {file_path}")

    df = pd.read_csv(file_path)
    
    # Generate LaTeX code using booktabs style for professional appearance
    latex_code = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        column_format='l' * len(df.columns),
        position='htbp',
        booktabs=True
    )
    return latex_code

@tool
def render_mermaid_diagram(mermaid_code: str, output_filename: str) -> str:
    """
    Converts Mermaid.js code into a PNG image using Mermaid-CLI (mmdc).
    Ideal for flowcharts, Gantt charts, and organizational structures.
    
    Args:
        mermaid_code: The raw Mermaid syntax string.
        output_filename: Desired filename (e.g., 'process_flow.png').
    """
    output_path = os.path.join("outputs/assets", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write mermaid code to temp file
    temp_mmd = "temp_diag.mmd"
    with open(temp_mmd, "w") as f:
        f.write(mermaid_code)
    
    # Execute Mermaid CLI (Requires mmdc installed via npm)
    try:
        subprocess.run(["mmdc", "-i", temp_mmd, "-o", output_path], check=True)
        os.remove(temp_mmd)
        return output_path
    except subprocess.CalledProcessError as e:
        if os.path.exists(temp_mmd): os.remove(temp_mmd)
        raise RuntimeError(f"Mermaid rendering failed. Ensure mmdc is installed: {e}")

@tool
def web_search_for_latest_data(query: str) -> List[Dict[str, str]]:
    """
    Performs a web search to find current market figures, news, or reference data.
    Helps prevent hallucination by providing grounded facts for the report.
    
    Args:
        query: The search query string.
    """
    # Note: In a real implementation, this would call APIs like Tavily, Serper, or Brave Search.
    # Returning a mock-up structure for demonstration.
    return [
        {"title": f"Report on {query}", "snippet": "Sample data found for the query...", "source": "https://example.com"}
    ]

# --- Helper to export tool definitions to LLM ---
def get_all_tool_schemas() -> List[Dict[str, Any]]:
    """Returns the list of tool definitions for LLM system prompt configuration."""
    return list(TOOL_REGISTRY.values())

# import functools
# import inspect
# import pandas as pd
# import os
# from typing import Any, Callable, Dict, List, Optional
# from pydantic import validate_arguments
# from src.common.schemas import ToolResponse, AssetType

# # ---------- 1. 도구 등록 및 메타데이터 추출 시스템 ----------

# class ToolRegistry:
#     """에이전트가 사용할 수 있는 도구들을 관리하는 레지스트리"""
#     def __init__(self):
#         self.tools: Dict[str, Dict[str, Any]] = {}

#     def register(self, func: Callable):
#         """함수를 도구로 등록하고 LLM용 스펙을 생성합니다."""
#         # Pydantic을 이용한 인자 유효성 검사 적용
#         validated_func = validate_arguments(func)
        
#         name = func.__name__
#         # Docstring을 LLM 설명서로 사용
#         description = inspect.getdoc(func) or "설명이 없는 도구입니다."
        
#         # 인자 정보 추출 (Type Hint 기반)
#         sig = inspect.signature(func)
#         parameters = {
#             k: str(v.annotation) for k, v in sig.parameters.items()
#         }

#         self.tools[name] = {
#             "func": validated_func,
#             "spec": {
#                 "name": name,
#                 "description": description,
#                 "parameters": parameters
#             }
#         }
#         return validated_func

# registry = ToolRegistry()

# def tool(func: Callable):
#     """도구 등록을 위한 데코레이터"""
#     return registry.register(func)

# # ---------- 2. 비즈니스 특화 핵심 도구 구현 ----------

# @tool
# def pandas_data_analyzer(file_path: str, query: Optional[str] = None) -> Dict[str, Any]:
#     """
#     CSV 파일을 읽어 데이터의 통계적 요약을 제공하고 LaTeX 표 코드를 생성합니다.
#     - file_path: 분석할 CSV 파일의 경로
#     - query: 특정 행을 필터링하기 위한 Pandas query 문장 (예: 'age > 30')
#     """
#     try:
#         df = pd.read_csv(file_path)
#         if query:
#             df = df.query(query)
            
#         summary = df.describe().to_dict()
#         # LaTeX 표 변환 (기본 포맷)
#         latex_table = df.to_latex(index=False, caption="Data Summary", label=f"tab:{os.path.basename(file_path)}")
        
#         return {
#             "summary": summary,
#             "latex_code": latex_table,
#             "row_count": len(df)
#         }
#     except Exception as e:
#         return {"error": f"Pandas 실행 오류: {str(e)}"}

# @tool
# def mermaid_diagram_generator(mermaid_code: str, output_name: str) -> str:
#     """
#     Mermaid 코드를 받아 프로세스 플로우차트나 조직도 이미지를 생성합니다.
#     - mermaid_code: 'graph TD; A-->B;' 형식의 Mermaid 문법 코드
#     - output_name: 저장될 이미지 파일명 (확장자 제외)
#     """
#     # 실제 구현 시에는 mmdc (Mermaid CLI) 등을 서브프로세스로 호출하거나 
#     # 외부 API를 사용하여 PNG/PDF로 렌더링합니다.
#     output_path = f"outputs/figures/{output_name}.png"
    
#     # (Placeholder) 실제 렌더링 로직이 들어갈 자리
#     # 예: subprocess.run(["mmdc", "-i", input_file, "-o", output_path])
    
#     return output_path

# @tool
# def business_web_search(query: str) -> List[Dict[str, str]]:
#     """
#     보고서 작성을 위해 최신 시장 지표, 수치, 뉴스 등을 검색합니다.
#     - query: 검색할 키워드 또는 질문
#     """
#     # RAG 시스템 또는 Serper/Tavily API와 연동하는 지점입니다.
#     # (Placeholder) 검색 결과 예시
#     return [
#         {"title": "2026 시장 전망", "snippet": "전년 대비 15% 성장 예상...", "source": "https://example.com/report"}
#     ]

# # ---------- 3. 도구 실행 핸들러 ----------

# def execute_tool(name: str, arguments: Dict[str, Any]) -> ToolResponse:
#     """이름을 기반으로 도구를 실행하고 표준화된 ToolResponse를 반환합니다."""
#     if name not in registry.tools:
#         return ToolResponse(success=False, output=None, observation="", error=f"도구 '{name}'을 찾을 수 없습니다.")
    
#     try:
#         func = registry.tools[name]["func"]
#         result = func(**arguments)
        
#         return ToolResponse(
#             success=True,
#             output=result,
#             observation=f"도구 {name} 실행이 완료되었습니다. 결과 요약: {str(result)[:200]}..."
#         )
#     except Exception as e:
#         return ToolResponse(success=False, output=None, observation="", error=str(e))