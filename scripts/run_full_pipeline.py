import argparse
import json
import time
import sys
import traceback
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from tqdm import tqdm

# Ensure pathing for local modules
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.common.state import DocumentGraph, NodeStatus, NodeType, DocumentNode
from src.common.schemas import PlannerInput, TemplateSpec
from src.agents.planner import PlannerAgent
from src.agents.text_agent import TextAgent
from src.agents.image_agent import ImageAgent
from src.agents.latex_integrator import LatexIntegratorAgent
from llm_client import DEFAULT_MODEL, LLMClient
from src.helpers.image_generator_sdxl import SDXLGenerator
from src.helpers.image_generator_flux import FluxGenerator

def get_client(clients, model_name):
    if model_name not in clients:
        clients[model_name] = LLMClient(model_name=model_name)
    return clients[model_name]

def build_user_query(item: Dict[str, Any]) -> str:
    # New-schema benchmark items (data/benchmark/benchmark_v1.jsonl) already carry
    # a single self-contained instruction; use it directly instead of the
    # legacy WikiHow-schema title/overview/instruction assembly below.
    natural_language_instruction = item.get("natural_language_instruction")
    if natural_language_instruction:
        return natural_language_instruction

    input_data = item.get("input", {})
    title = input_data.get("title", "")
    overview = input_data.get("overview", "")
    instruction = item.get("instruction", "")
    return "\n\n".join([
        f"Title: {title}",
        f"Overview: {overview}",
        f"Instruction: {instruction}",
    ])

class DocGenPipeline:
    def __init__(self, 
                 planner_model: str,
                 text_model: str,
                 image_model: str,
                 integrator_model: str,
                 output_dir: str = "outputs/generations"):
        
        if image_model == 'sdxl':
            image_generator = SDXLGenerator()
        elif image_model == 'flux':
            image_generator = FluxGenerator()

        clients = {}
        agents = {
            'planner_agent': PlannerAgent(llm_client=get_client(clients, planner_model)),
            'text_agent': TextAgent(llm_client=get_client(clients, text_model)),
            'integrator': LatexIntegratorAgent(llm_client=get_client(clients, integrator_model)),
            'image_agent': ImageAgent(llm_client=get_client(clients, text_model), image_generator=image_generator)
        }

        self.planner = agents['planner_agent']
        self.text_agent = agents['text_agent']
        self.image_agent = agents['image_agent']
        self.integrator = agents['integrator']

        self.output_base = Path(output_dir)

    def run(self, user_query: str, request_id: str, iterations: int, template_name: str = "default"):
        project_dir = self.output_base / request_id
        project_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Starting Pipeline for Request: {request_id}")

        # 1. Initialize Shared Document Graph via Planner 
        graph = DocumentGraph()
        planner_in = PlannerInput(
            request_id=request_id,
            user_query=user_query,
            template=TemplateSpec(name=template_name, main_tex_template="main.tex")
        )
        
        print("[Pipeline] Phase 1: Planning Document Structure...")
        plan = self.planner.generate_plan(planner_in)
        with open(project_dir / "plan.json", "w") as f:
            f.write(plan.model_dump_json(indent=2))
        
        # Construct the Graph from Planner Output 
        for node_data in plan.nodes_to_create:
            node = DocumentNode(**node_data)
            graph.add_node(node)
        for parent, child in plan.hierarchy_edges:
            graph.add_edge(parent, child)
        graph.set_blueprint(plan.blueprint)
        for section_layout in plan.blueprint.sections:
            graph.nodes[section_layout.section_id].layout_block_ids = [
                block.id for block in section_layout.blocks
            ]

        # 2. Execution Loop: Process nodes until all are DRAFTED 
        max_iterations = iterations
        for i in range(max_iterations):
            pending_nodes = [n for n in graph.nodes.values() if n.status in [NodeStatus.PENDING, NodeStatus.ERROR]]
            
            if not pending_nodes:
                print("[Pipeline] All nodes drafted successfully.")
                break
                
            print(f"[Pipeline] Iteration {i+1}: Processing {len(pending_nodes)} nodes...")
            
            for node in pending_nodes:
                if node.type in [NodeType.SECTION, NodeType.SUBSECTION]:
                    self.text_agent.process_node(node.id, graph, request_id, output_dir=str(self.output_base))
                elif node.type == NodeType.FIGURE:
                    self.image_agent.process_figure_node(node.id, graph, request_id, output_dir=str(self.output_base))
                # Tables are handled inside TextAgent via tools

        # Phase 3 reads each SECTION/SUBSECTION's persisted blocks.json and each
        # FIGURE's registered asset off disk -- if a node never reached DRAFTED
        # (its content agent kept failing every iteration, most often because a
        # blueprint instruction asks for something its block kind structurally
        # can't render, e.g. a numbered/labeled equation inside a `definition`
        # block), that artifact was never written. Failing loudly here, with
        # every unfinished node's last error, beats the alternative: silently
        # falling into Phase 3 and hitting a FileNotFoundError several stack
        # frames deep in LatexIntegratorAgent that doesn't say which content
        # agent actually failed or why.
        unfinished = [
            node for node in graph.nodes.values()
            if node.type in (NodeType.SECTION, NodeType.SUBSECTION, NodeType.FIGURE)
            and node.status != NodeStatus.DRAFTED
        ]
        if unfinished:
            details = "; ".join(
                f"{node.id} ({node.status.value}: {node.last_error})" for node in unfinished
            )
            raise RuntimeError(
                f"{len(unfinished)} node(s) never finished drafting after {max_iterations} "
                f"iteration(s), cannot start LaTeX integration: {details}"
            )

        # 3. Integration & Self-Correcting Compilation
        print("[Pipeline] Phase 3: Integrating Assets and Compiling...")
        # success = self.integrator.integrate_and_compile(graph, str(project_dir))
        success = self.integrator.integrate(request_id, graph, output_dir=str(self.output_base))

        if success:
            print(f">>> Success! Final PDF located in: {project_dir}")
        else:
            print(">>> Pipeline finished with compilation warnings/errors.")

def run_batch(pipeline: DocGenPipeline, dataset_path: str, iterations: int):
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    for index, item in enumerate(tqdm(items, desc="Batch")):
        # New-schema items carry a stable sample_id (e.g. "math_real_analysis_0001");
        # fall back to the legacy stress_factor grouping when it's absent.
        group = item.get("sample_id") or item.get("stress_factor", "unknown")
        request_id = f"{group}/sample_{index:03d}"
        try:
            pipeline.run(
                user_query=build_user_query(item),
                request_id=request_id,
                iterations=iterations,
            )
        except Exception as exc:
            print(f"[Batch] Error at index {index} ({request_id}): {exc}")
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="MALaDocGen: Advanced Multi-Agent LaTeX Generator")
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("-q", "--query", help="User request for the document")
    query_group.add_argument("--dataset", help="Path to a JSONL dataset for batch generation")
    parser.add_argument("-id", "--request-id", default=f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Unique request ID (single-query mode only)")
    parser.add_argument("--planner-model", default=DEFAULT_MODEL, help="LLM for the Planner Agent, e.g. gpt-5.4-mini or gemini-3-flash")
    parser.add_argument("--text-model", default=DEFAULT_MODEL, help="LLM for the Text Generation Agent, e.g. gpt-5.4-mini or gemini-3-flash")
    parser.add_argument("--image-model", default="flux", choices=['sdxl', 'flux'], help="Model for the Image Strategy Agent")
    parser.add_argument("--integrator-model", default=DEFAULT_MODEL, help="LLM for the LaTeX Integrator Agent, e.g. gpt-5.4-mini or gemini-3-flash")
    parser.add_argument("--out-dir", default="outputs/generations", help="Output directory")
    parser.add_argument("--iterations", type=int, default=5, help="Maximum number of iterations")
    args = parser.parse_args()

    out_dir = args.out_dir
    if args.dataset:
        # Stamp each batch run into its own directory instead of overwriting
        # the previous batch's outputs at the same --out-dir path.
        timestamp = datetime.now().strftime("%m%d%H%M")
        out_dir = f"{out_dir.rstrip('/')}_{timestamp}"
        print(f"[Batch] Writing outputs to: {out_dir}")

    pipeline = DocGenPipeline(
        planner_model=args.planner_model,
        text_model=args.text_model,
        image_model=args.image_model,
        integrator_model=args.integrator_model,
        output_dir=out_dir
    )

    if args.dataset:
        run_batch(pipeline, dataset_path=args.dataset, iterations=args.iterations)
    else:
        pipeline.run(user_query=args.query, request_id=args.request_id, iterations=args.iterations)

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Total duration: {time.perf_counter() - start:.4f} seconds")
