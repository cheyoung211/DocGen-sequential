import argparse
import json
import time
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from tqdm import tqdm

# Ensure pathing for local modules
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.common.state import DocumentGraph, NodeStatus, NodeType, DocumentNode
from src.common.schemas import PlannerInput, TemplateSpec
from src.agents.planner import DEFAULT_ENABLED_PLANNER_VALIDATORS, PlannerAgent
from src.agents.text_agent import DEFAULT_ENABLED_VALIDATORS, TextAgent
from src.agents.image_agent import ImageAgent
from src.agents.latex_assembler import DEFAULT_ENABLED_ASSEMBLER_VALIDATORS, LatexAssembler
from src.agents.latex_compiler import LatexCompiler
from src.dataset.schemas import BenchmarkItem
from src.evaluation.compile_log_parser import classify_recovery_action, extract_error_locations
from src.evaluation.event_logger import AttemptTracker, EventLogger
from src.evaluation.metrics.contract_satisfaction import evaluate_contract
from src.evaluation.schemas import RecoveryAction, RunResult, Stage, TokenUsage
from llm_client import DEFAULT_MODEL, LLMClient
from src.helpers.image_generator_sdxl import SDXLGenerator
from src.helpers.image_generator_flux import FluxGenerator

def get_client(clients, model_name, seed=None):
    if model_name not in clients:
        clients[model_name] = LLMClient(model_name=model_name, seed=seed)
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

def _build_run_result(
    *,
    request_id: str,
    status: str,
    error: Optional[BaseException],
    event_logger: EventLogger,
    start_time: Optional[float],
    token_usage_records: Optional[List[dict]],
    benchmark_item: Optional[BenchmarkItem],
    producer_models: Optional[Dict[str, str]],
    project_dir: Path,
    seed: Optional[int] = None,
) -> RunResult:
    """Assemble the full evaluation-layer RunResult for one request.

    contract_result is computed best-effort even on a failure path: a
    partially-drafted document (e.g. only 2 of 5 sections got that far
    before a later one hit content_generation_error) is still informative
    for Metric 7, and evaluate_contract() degrades to a null-filled result
    on its own if plan.json never got written at all.
    """
    compile_result = event_logger.compile_result
    contract_result = evaluate_contract(project_dir, benchmark_item) if benchmark_item else None

    token_usage = None
    if token_usage_records:
        token_usage = TokenUsage(
            total_input_tokens=sum(r.get("input_tokens", 0) for r in token_usage_records),
            total_output_tokens=sum(r.get("output_tokens", 0) for r in token_usage_records),
            total_tokens=sum(r.get("total_tokens", 0) for r in token_usage_records),
            call_count=len(token_usage_records),
        )

    return RunResult(
        run_id=request_id,
        benchmark_id=benchmark_item.sample_id if benchmark_item else None,
        producer_model=(producer_models or {}).get("text", "unknown"),
        producer_models=producer_models or {},
        seed=seed,
        completed=(status == "success"),
        status=status,
        error_type=type(error).__name__ if error is not None else None,
        error_message=str(error) if error is not None else None,
        compile_success=bool(compile_result and compile_result.compile_success),
        compile_result=compile_result,
        verification_events=event_logger.verification_events,
        recovery_events=event_logger.recovery_events,
        contract_result=contract_result,
        total_generation_attempts=event_logger.total_generation_attempts,
        total_recovery_attempts=event_logger.total_recovery_attempts,
        token_usage=token_usage,
        elapsed_time=(time.perf_counter() - start_time) if start_time is not None else None,
    )


def _write_run_log(
    project_dir: Path,
    request_id: str,
    status: str,
    error: Optional[BaseException] = None,
    compile_log_path: Optional[Path] = None,
    *,
    event_logger: Optional[EventLogger] = None,
    start_time: Optional[float] = None,
    token_usage_records: Optional[List[dict]] = None,
    benchmark_item: Optional[BenchmarkItem] = None,
    producer_models: Optional[Dict[str, str]] = None,
    seed: Optional[int] = None,
) -> None:
    """Persist one request's outcome to ``<project_dir>/run_result.json``.

    Written for every request -- success, a content-generation failure
    (planning or drafting never reached DRAFTED for every node), an
    integration failure (blueprint assembly / pre-compile validation, which
    raises before pdflatex ever runs, so no compile.log exists yet), or a
    compile failure (pdflatex/tectonic ran and failed; compile.log exists
    and its content is embedded here rather than left as a separate file a
    batch summary would otherwise have to go find and open itself).

    When ``event_logger`` is supplied, the file's top-level shape becomes a
    full evaluation-layer ``RunResult`` instead (so ``src/evaluation/
    aggregate.py`` can parse it directly) -- assembly is wrapped in its own
    try/except so a bug in this new instrumentation can never prevent the
    base outcome record from being written, or break an otherwise-successful
    run. With no new keyword arguments passed (today's only call shape),
    this function's output is unchanged.
    """
    if event_logger is not None:
        try:
            run_result = _build_run_result(
                request_id=request_id, status=status, error=error, event_logger=event_logger,
                start_time=start_time, token_usage_records=token_usage_records,
                benchmark_item=benchmark_item, producer_models=producer_models,
                project_dir=project_dir, seed=seed,
            )
            project_dir.mkdir(parents=True, exist_ok=True)
            (project_dir / "run_result.json").write_text(
                run_result.model_dump_json(indent=2), encoding="utf-8"
            )
            return
        except Exception as exc:
            print(f"[Pipeline] Warning: evaluation-layer RunResult assembly failed, "
                  f"falling back to the legacy run_result.json shape: {exc}")

    record: Dict[str, Any] = {
        "request_id": request_id,
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if error is not None:
        record["error_type"] = type(error).__name__
        record["error_message"] = str(error)
    if compile_log_path is not None:
        record["compile_log"] = (
            compile_log_path.read_text(encoding="utf-8", errors="replace")
            if compile_log_path.exists()
            else None
        )
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "run_result.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _warn_verifier_pairing_mismatches(
    text_agent_validators: Dict[str, bool], integrator_validators: Dict[str, bool]
) -> None:
    """Flag a code that's on in one layer and off in the other.

    TextAgent's forbidden_environment/manual_label/table_columns checks and
    LatexAssembler's copies of the same three checks are independently
    toggleable (see DEFAULT_ENABLED_VALIDATORS / DEFAULT_ENABLED_ASSEMBLER_
    VALIDATORS) -- deliberately not mirrored, so a verifier ablation run can
    study either layer alone. This just warns when that's likely accidental:
    a check "disabled" for the run that's still enforced by the other layer.
    """
    shared_codes = set(text_agent_validators) & set(integrator_validators)
    for code in sorted(shared_codes):
        text_state = text_agent_validators[code]
        integrator_state = integrator_validators[code]
        if text_state != integrator_state:
            print(
                f"[Pipeline] WARNING: '{code}' verifier mismatch -- "
                f"TextAgent={'ON' if text_state else 'OFF'}, "
                f"LatexAssembler={'ON' if integrator_state else 'OFF'}."
            )


class DocGenPipeline:
    def __init__(self,
                 planner_model: str,
                 text_model: str,
                 image_model: str,
                 integrator_model: str,
                 output_dir: str = "outputs/generations",
                 text_agent_enabled_validators: Optional[Dict[str, bool]] = None,
                 planner_enabled_validators: Optional[Dict[str, bool]] = None,
                 assembler_enabled_validators: Optional[Dict[str, bool]] = None,
                 max_compile_repair_attempts: int = 2,
                 seed: Optional[int] = None):

        # 'none' disables image generation entirely for this run: no image
        # model is loaded, and the planner is instructed (and validated) to
        # never create a FIGURE node, so ImageAgent is never invoked.
        self.images_enabled = image_model != 'none'

        # Retained for RunResult.producer_models -- this pipeline allows 4
        # independently-chosen models per run, so a single flat model-name
        # string would lose real information for a mixed-model run.
        self.planner_model = planner_model
        self.text_model = text_model
        self.image_model = image_model
        self.integrator_model = integrator_model
        # Recorded into RunResult so a run_result.json can be traced back to
        # the seed that produced it -- see get_client()/LLMClient for what
        # this actually does per provider (Gemini: real seed; OpenAI: no
        # seed support, forced temperature=0 instead).
        self.seed = seed

        clients = {}
        agents = {
            'planner_agent': PlannerAgent(
                llm_client=get_client(clients, planner_model, seed=seed),
                enabled_validators=planner_enabled_validators,
            ),
            'text_agent': TextAgent(
                llm_client=get_client(clients, text_model, seed=seed),
                enabled_validators=text_agent_enabled_validators,
            ),
            'assembler': LatexAssembler(
                llm_client=get_client(clients, integrator_model, seed=seed),
                enabled_validators=assembler_enabled_validators,
            ),
            'compiler': LatexCompiler(
                llm_client=get_client(clients, integrator_model, seed=seed),
            ),
        }
        _warn_verifier_pairing_mismatches(
            agents['text_agent'].enabled_validators, agents['assembler'].enabled_validators
        )
        if self.images_enabled:
            if image_model == 'sdxl':
                image_generator = SDXLGenerator()
            elif image_model == 'flux':
                image_generator = FluxGenerator()
            agents['image_agent'] = ImageAgent(llm_client=get_client(clients, text_model, seed=seed), image_generator=image_generator)
        else:
            agents['image_agent'] = None

        self.planner = agents['planner_agent']
        self.text_agent = agents['text_agent']
        self.image_agent = agents['image_agent']
        self.assembler = agents['assembler']
        self.compiler = agents['compiler']

        self.output_base = Path(output_dir)
        # Bounds LatexCompiler-triggered regeneration rounds (see
        # _repair_and_recompile) -- distinct from `iterations` (Phase 2's
        # first-draft retry budget) since a compile-repair round is much
        # cheaper (usually one block, not a whole section) and this backstop
        # exists mainly to prevent looping forever against a section that
        # keeps re-failing in a new way each time.
        self.max_compile_repair_attempts = max_compile_repair_attempts

    def run(
        self,
        user_query: str,
        request_id: str,
        iterations: int,
        template_name: str = "default",
        benchmark_item: Optional[BenchmarkItem] = None,
    ):
        project_dir = self.output_base / request_id
        project_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Starting Pipeline for Request: {request_id}")

        start_time = time.perf_counter()
        event_logger = EventLogger(run_id=request_id)
        token_usage_records: List[dict] = []
        producer_models = {
            "planner": self.planner_model,
            "text": self.text_model,
            "image": self.image_model,
            "integrator": self.integrator_model,
        }

        def _write_log(
            status: str,
            error: Optional[BaseException] = None,
            compile_log_path: Optional[Path] = None,
        ) -> None:
            _write_run_log(
                project_dir, request_id, status=status, error=error, compile_log_path=compile_log_path,
                event_logger=event_logger, start_time=start_time, token_usage_records=token_usage_records,
                benchmark_item=benchmark_item, producer_models=producer_models, seed=self.seed,
            )

        # 1. Initialize Shared Document Graph via Planner
        graph = DocumentGraph()
        try:
            planner_in = PlannerInput(
                request_id=request_id,
                user_query=user_query,
                template=TemplateSpec(name=template_name, main_tex_template="main.tex"),
                allow_figures=self.images_enabled,
                required_sections=benchmark_item.required_sections if benchmark_item else [],
            )

            print("[Pipeline] Phase 1: Planning Document Structure...")
            plan = self.planner.generate_plan(
                planner_in, event_logger=event_logger, usage_sink=token_usage_records
            )
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
                        self.text_agent.process_node(
                            node.id, graph, request_id, output_dir=str(self.output_base),
                            event_logger=event_logger, usage_sink=token_usage_records,
                            outer_iteration=i + 1,
                        )
                    elif node.type == NodeType.FIGURE:
                        self.image_agent.process_figure_node(
                            node.id, graph, request_id, output_dir=str(self.output_base),
                            event_logger=event_logger, usage_sink=token_usage_records,
                            outer_iteration=i + 1,
                        )
                    # Tables are handled inside TextAgent via tools

            # Phase 3 reads each SECTION/SUBSECTION's persisted blocks.json and each
            # FIGURE's registered asset off disk -- if a node never reached DRAFTED
            # (its content agent kept failing every iteration, most often because a
            # blueprint instruction asks for something its block kind structurally
            # can't render, e.g. a numbered/labeled equation inside a `definition`
            # block), that artifact was never written. Failing loudly here, with
            # every unfinished node's last error, beats the alternative: silently
            # falling into Phase 3 and hitting a FileNotFoundError several stack
            # frames deep in LatexAssembler that doesn't say which content
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
        except Exception as exc:
            _write_log("content_generation_error", error=exc)
            raise

        # 3. Integration & Self-Correcting Compilation
        print("[Pipeline] Phase 3: Integrating Assets and Compiling...")
        try:
            self.assembler.assemble(
                request_id, graph, output_dir=str(self.output_base), event_logger=event_logger
            )
        except Exception as exc:
            # Raised by LatexAssembler.assemble() itself (blueprint assembly /
            # pre-compile validation) -- before pdflatex ever runs, so no
            # compile.log exists yet to attach.
            _write_log("integration_error", error=exc)
            raise

        success = self.compiler.compile_pdf(
            request_id, output_dir=str(self.output_base), event_logger=event_logger
        )

        if not success:
            success = self._repair_and_recompile(
                request_id, graph, event_logger=event_logger, token_usage_records=token_usage_records,
            )

        if success:
            print(f">>> Success! Final PDF located in: {project_dir}")
            _write_log("success")
        else:
            print(">>> Pipeline finished with compilation warnings/errors.")
            _write_log("compile_error", compile_log_path=project_dir / "compile.log")

    def _repair_and_recompile(
        self,
        request_id: str,
        graph: DocumentGraph,
        *,
        event_logger: EventLogger,
        token_usage_records: List[dict],
    ) -> bool:
        """React to a failed compile_pdf() by regenerating exactly the block
        (or, failing that, the whole section) the compiler pointed at, then
        re-assembling and recompiling -- instead of giving up after the
        first failed compile the way this pipeline always has.

        Every location tectonic reports is against the physical file it was
        reading (``sections/{section_id}.tex``, one per graph section -- see
        ``LatexAssembler._write_main_tex``), so the section is known for
        free; the block within it is resolved via the
        ``% <layout-anchor:...>`` comment ``LatexAssembler._render_section``
        writes ahead of each block (``LatexAssembler.locate_block_at_line``).
        When no block resolves -- the error sits before the first anchor, or
        the error type isn't one TextAgent content can fix at all (a missing
        asset file, a broken document frame) -- the whole section is
        redrafted via the existing ``process_node`` instead, or the loop
        stops outright for a non-content error type.

        One AttemptTracker spans the whole repair loop, not one per
        recompile, so a fix that takes 2 rounds still pairs as a single
        failure -> recovery outcome -- the same semantics Planner/TextAgent's
        own inner retry loops already have (see AttemptTracker's docstring).
        """
        project_dir = self.output_base / request_id
        compile_log_path = project_dir / "compile.log"
        tracker: Optional[AttemptTracker] = None

        for attempt in range(1, self.max_compile_repair_attempts + 1):
            log_text = (
                compile_log_path.read_text(encoding="utf-8", errors="replace")
                if compile_log_path.is_file() else ""
            )
            engine_name = event_logger.compile_result.engine if event_logger.compile_result else None
            locations = extract_error_locations(log_text, engine_name or self.compiler.engine)
            regenerable = [
                loc for loc in locations
                if classify_recovery_action(loc.error_type) == RecoveryAction.REGENERATE_CONTENT
            ]
            if not regenerable:
                # Nothing here is a content problem TextAgent can fix
                # (missing asset, broken document frame, or no location at
                # all) -- stop and let the caller report the failure as before.
                break
            location = regenerable[0]
            if graph.nodes.get(location.section_id) is None:
                break

            if tracker is None:
                tracker = AttemptTracker(
                    event_logger,
                    stage=Stage.COMPILE,
                    verifier="latex_compiler",
                    recovery_action=RecoveryAction.REGENERATE_CONTENT.value,
                    artifact_id=request_id,
                    section_id=location.section_id,
                    producer_model=self.text_model,
                )
            tracker.record_failure(attempt, location.error_type, location.message)

            section_tex_path = project_dir / "sections" / f"{location.section_id}.tex"
            block_id = LatexAssembler.locate_block_at_line(section_tex_path, location.line)
            print(
                f"[Pipeline] Compile repair {attempt}/{self.max_compile_repair_attempts}: "
                f"{location.section_id}" + (f" block '{block_id}'" if block_id else " (whole section)") +
                f" -- {location.message}"
            )

            try:
                if block_id:
                    self.text_agent.regenerate_blocks(
                        location.section_id, block_id, location.message, graph, request_id,
                        output_dir=str(self.output_base), event_logger=event_logger,
                        usage_sink=token_usage_records,
                    )
                else:
                    self.text_agent.process_node(
                        location.section_id, graph, request_id,
                        output_dir=str(self.output_base), event_logger=event_logger,
                        usage_sink=token_usage_records,
                    )
            except Exception as exc:
                print(f"[Pipeline] Compile repair regeneration failed: {exc}")
                break

            if graph.nodes[location.section_id].status != NodeStatus.DRAFTED:
                # regenerate_blocks()/process_node() already marked the node
                # ERROR and logged its own failure -- nothing new to compile.
                break

            try:
                self.assembler.assemble(
                    request_id, graph, output_dir=str(self.output_base), event_logger=event_logger
                )
            except Exception as exc:
                print(f"[Pipeline] Re-assembly after compile repair failed: {exc}")
                break

            if self.compiler.compile_pdf(
                request_id, output_dir=str(self.output_base), event_logger=event_logger
            ):
                tracker.record_success(attempt)
                return True

        if tracker is not None:
            tracker.record_exhausted()
        return False

def run_batch(pipeline: DocGenPipeline, dataset_path: str, iterations: int):
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    for index, item in enumerate(tqdm(items, desc="Batch")):
        # New-schema items carry a stable sample_id (e.g. "math_real_analysis_0001");
        # fall back to the legacy stress_factor grouping when it's absent.
        group = item.get("sample_id") or item.get("stress_factor", "unknown")
        request_id = f"{group}/sample_{index:03d}"
        try:
            benchmark_item = BenchmarkItem.model_validate(item)
        except Exception:
            # Legacy WikiHow-schema rows (or any dict not shaped like the
            # current BenchmarkItem contract) degrade to no contract
            # checking rather than crashing the batch.
            benchmark_item = None
        try:
            pipeline.run(
                user_query=build_user_query(item),
                request_id=request_id,
                iterations=iterations,
                benchmark_item=benchmark_item,
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
    parser.add_argument("--image-model", default="none", choices=['sdxl', 'flux', 'none'], help="Model for the Image Strategy Agent, or 'none' to disable image generation for this run (default)")
    parser.add_argument("--integrator-model", default=DEFAULT_MODEL, help="LLM slot shared by LatexAssembler/LatexCompiler (currently unused by either), e.g. gpt-5.4-mini or gemini-3-flash")
    parser.add_argument("--out-dir", default="outputs/generations", help="Output directory")
    parser.add_argument("--iterations", type=int, default=3, help="Maximum number of iterations")
    parser.add_argument(
        "--max-compile-repair-attempts", type=int, default=2,
        help="Max compile-failure repair rounds (regenerate the flagged block/section, "
             "re-assemble, recompile) before giving up (default 2).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Fix the LLM sampling seed for every agent's LLMClient in this run, for "
             "reproducibility across runs on the same data. Gemini models honor a real "
             "seed (best effort, per Gemini's own docs). OpenAI models are called via "
             "the Responses API, which has no seed parameter at all -- passing --seed "
             "forces temperature=0 for OpenAI calls instead, as the closest available "
             "approximation; this is not a guaranteed-identical-output determinism, only "
             "a reduction in sampling randomness.",
    )
    parser.add_argument(
        "--disable-text-validator",
        action="append",
        choices=sorted(DEFAULT_ENABLED_VALIDATORS),
        metavar="CODE",
        help=(
            "Turn off one TextAgent content-quality verifier by code, for "
            "verifier ablation experiments (repeatable). Structural checks "
            "(block id/order/type, non-empty content) are never disabled. "
            f"Codes: {', '.join(sorted(DEFAULT_ENABLED_VALIDATORS))}"
        ),
    )
    parser.add_argument(
        "--disable-planner-validator",
        action="append",
        choices=sorted(DEFAULT_ENABLED_PLANNER_VALIDATORS),
        metavar="CODE",
        help=(
            "Turn off one Planner content-quality verifier by code, for "
            "verifier ablation experiments (repeatable). Referential-integrity "
            "checks (duplicate/unknown ids, empty section blocks, orphan "
            "figures, figure slot mismatches, ...) are never disabled -- the "
            "composer cannot build a document without them holding. "
            f"Codes: {', '.join(sorted(DEFAULT_ENABLED_PLANNER_VALIDATORS))}"
        ),
    )
    parser.add_argument(
        "--disable-assembler-validator",
        action="append",
        choices=sorted(DEFAULT_ENABLED_ASSEMBLER_VALIDATORS),
        metavar="CODE",
        help=(
            "Turn off one LatexAssembler verifier by code, for verifier "
            "ablation experiments (repeatable). forbidden_environment/"
            "manual_label/table_columns are the composer's own "
            "second-line-of-defense copy of the same check TextAgent runs "
            "first -- toggled independently, not mirrored from "
            "--disable-text-validator. The rest (layout_policy, "
            "figure_asset_completeness, figure_single_use, reference, "
            "document_frame) gate the composer's own pre-compile structural "
            "checks. Block id/order/type/asset_id consistency is never "
            "disabled -- the renderer assumes it already holds. "
            f"Codes: {', '.join(sorted(DEFAULT_ENABLED_ASSEMBLER_VALIDATORS))}"
        ),
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    if args.dataset:
        # Stamp each batch run into its own directory instead of overwriting
        # the previous batch's outputs at the same --out-dir path.
        timestamp = datetime.now().strftime("%m%d%H%M")
        out_dir = f"{out_dir.rstrip('/')}_{timestamp}"
        print(f"[Batch] Writing outputs to: {out_dir}")

    text_agent_enabled_validators = (
        {code: False for code in args.disable_text_validator} if args.disable_text_validator else None
    )
    if text_agent_enabled_validators:
        print(f"[Pipeline] TextAgent verifiers disabled for this run: {sorted(text_agent_enabled_validators)}")

    planner_enabled_validators = (
        {code: False for code in args.disable_planner_validator} if args.disable_planner_validator else None
    )
    if planner_enabled_validators:
        print(f"[Pipeline] Planner verifiers disabled for this run: {sorted(planner_enabled_validators)}")

    assembler_enabled_validators = (
        {code: False for code in args.disable_assembler_validator}
        if args.disable_assembler_validator else None
    )
    if assembler_enabled_validators:
        print(f"[Pipeline] Assembler verifiers disabled for this run: {sorted(assembler_enabled_validators)}")

    pipeline = DocGenPipeline(
        planner_model=args.planner_model,
        text_model=args.text_model,
        image_model=args.image_model,
        integrator_model=args.integrator_model,
        output_dir=out_dir,
        text_agent_enabled_validators=text_agent_enabled_validators,
        planner_enabled_validators=planner_enabled_validators,
        assembler_enabled_validators=assembler_enabled_validators,
        max_compile_repair_attempts=args.max_compile_repair_attempts,
        seed=args.seed,
    )

    if args.dataset:
        run_batch(pipeline, dataset_path=args.dataset, iterations=args.iterations)
    else:
        pipeline.run(user_query=args.query, request_id=args.request_id, iterations=args.iterations)

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Total duration: {time.perf_counter() - start:.4f} seconds")
