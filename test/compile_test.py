import subprocess
import shutil
import os

def compile_latex_with_tectonic(tex_path: str):
    """
    tectonic으로 LaTeX 문서를 컴파일하고,
    성공 여부(True/False)와 컴파일 로그를 문자열로 반환.
    """
    if shutil.which("tectonic") is None:
        return False, "tectonic이 설치되어 있지 않습니다. `conda install -c conda-forge tectonic`으로 설치하세요."

    tex_dir = os.path.dirname(tex_path)
    tex_file = os.path.basename(tex_path)

    cmd = ["tectonic", tex_file]

    result = subprocess.run(
        cmd,
        cwd=tex_dir if tex_dir else None,  # tex_path가 상대경로/절대경로 어느 쪽이든 대응
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    success = (result.returncode == 0)
    logs = result.stdout + "\n" + result.stderr
    return success, logs


if __name__ == "__main__":
    tex_path = "/data1/home/chelsy211/document_generation/outputs/req_20260224_164649/main.tex"  # 여기를 실제 main.tex 경로로 변경
    success, logs = compile_latex_with_tectonic(tex_path)

    if success:
        print("✅ 컴파일 성공")
    else:
        print("❌ 컴파일 실패")

    print("=== 컴파일 로그 ===")
    print(logs)