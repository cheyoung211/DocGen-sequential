# Document Generation API

All language-model agents now call OpenAI's Responses API with `gpt-4o-mini` by default. The local Qwen/Transformers model, tokenizer, CUDA placement, and 4-bit quantization code have been removed from the active LLM path. SDXL and Flux remain optional local image renderers; they are not language models.

## Setup

```bash
cd document_generation_api
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY="your_openai_api_key"
```

Alternatively, copy `.env.example` into your preferred secret-management workflow. The code deliberately reads only `OPENAI_API_KEY` and never writes it to output metadata.

## Run

```bash
python3 scripts/run_full_pipeline.py \
  --query "Create a technical report about zero-trust security." \
  --image-model flux
```

The planner, text writer, image-prompt strategist, and LaTeX integrator all default to `gpt-4o-mini`. The `--planner-model`, `--text-model`, and `--integrator-model` options remain available only when an explicit OpenAI model override is required.

For a minimal connectivity check after setting the key:

```bash
python3 test/test_openai.py
```

## Benchmark Dataset (LaTeX-Focused Evaluation Set)

`data/benchmark/benchmark_v1.jsonl` is an **evaluation-only** benchmark (never used for training) of 120 long-form
document-generation tasks across four domains -- mathematics, statistics/ML, scientific/engineering, and
financial/economic analysis -- chosen because they genuinely require LaTeX: theorem/definition/proof environments,
numbered equations, notation tables, and cross-referenced sections. It replaces the earlier WikiHow-derived
benchmark (`dataset/processed/wikihow_*`, `src/dataset/prompt_conversion.py`), which under-used LaTeX.

Each line is a full document *contract* (required sections, required LaTeX components, hard constraints,
cross-section dependencies, notation/terminology constraints, and a `natural_language_instruction` that encodes all
of the above) -- not a gold reference document. See `src/dataset/schemas.py` for the full schema.

### Rebuild the dataset

```bash
python3 -m pip install -r requirements.txt
python3 -m src.dataset.build_dataset --config configs/dataset_v1.yaml --output-dir data --seed 42
```

This is fully offline and deterministic by default (`network_sources_enabled: false` in
`configs/dataset_v1.yaml`): all 120 items come from the hand-curated seed bank in
`src/dataset/source_adapters/manual_seed.py`. Re-running with the same `--seed` reproduces
`data/benchmark/benchmark_v1.jsonl` byte-for-byte.

Output layout (spec section 10):

```
data/
  normalized_seeds/<domain>.jsonl       # Stage 1 seed records
  benchmark/
    benchmark_v1.jsonl, benchmark_v1.json
    dev_debug.jsonl                     # 8-12 items for smoke-testing consumers only, not a training split
    splits/<domain>.jsonl               # per-domain subsets
  metadata/
    source_registry.json                # every SourceRecord used, with license metadata
    license_report.json
    dataset_statistics.json
    rejected_items.jsonl                # items that failed validators.py, with reasons
```

### Run the dataset's test suite

```bash
python3 -m pip install pytest
python3 -m pytest test/dataset/ -v
```

Covers schema validation, the 15 quality validators (`src/dataset/validators.py`), structural rules per difficulty
tier for every seed in the bank (`src/dataset/instruction_builder.py`), the source adapters against mocked API
responses (`src/dataset/source_adapters/*`), and an end-to-end `build_dataset()` smoke test.

### Using the benchmark with the existing run scripts

`scripts/run_full_pipeline.py --dataset data/benchmark/benchmark_v1.jsonl` and
`src/baseline_single_llm/single_llm.py -d data/benchmark/benchmark_v1.jsonl` both already understand the new
schema: `build_user_query` / `build_user_prompt` use `natural_language_instruction` directly when present, and fall
back to the legacy WikiHow `input.title` / `input.overview` / `instruction` fields otherwise, so old and new
datasets both work unmodified.

### Licensing assumptions and unresolved risks

- **Manual-seed items (all 120 committed items) are Tier 3**: hand-curated topic/subtopic/learning-objective seeds
  based on standard curricula, containing no reused copyrighted prose. Each cites a `standard_curriculum`
  `SourceRecord` (`license_name: "Standard curriculum (no copyrighted prose reused)"`), which is on the registry's
  approved-license allowlist (`src/dataset/source_registry.py`) and needs no manual review.
- **OpenStax/NASA/FRED adapters are real but off by default** (`network_sources_enabled: false`). If enabled, they
  fetch *metadata only* (book titles, dataset titles, series titles) -- never prose -- but:
  - The OpenStax adapter cannot confirm a specific book edition's exact license from the CMS API's `books.Book`
    fields alone, so every `SourceRecord` it produces is force-flagged `manual_review_required=True` until a human
    confirms the edition's license on `openstax.org` directly.
  - The NASA adapter records whatever `license_title` the CKAN API returns; entries with `"other-license-specified"`
    or missing license titles are auto-flagged for review by `SourceRegistry._apply_allowlist`.
  - The FRED adapter needs `FRED_API_KEY` (never hard-coded; read from the environment) and always flags its
    records for manual review, since FRED's terms can change independently of this codebase.
- **arXiv/SEC/World Bank adapters are stubs** (`src/dataset/source_adapters/{arxiv_metadata,sec_metadata,
  world_bank_metadata}.py`) -- they return no seeds. Each module's docstring documents the intended endpoint and
  TODOs for a v1.1 implementation.
- **Unresolved risk**: the `natural_language_instruction` "semantic equivalence" check (spec section 6) is
  implemented as deterministic keyword/text coverage (`validators.py` rules 9-10), not true NLU-level semantic
  verification -- acceptable here because `instruction_builder.py` *renders* the instruction from the same
  structured fields the checker inspects, so coverage holds by construction, but this checker would not by itself
  catch a mismatch introduced by a future hand-edit of `natural_language_instruction` after generation.
- **Unresolved risk**: the current multi-stage generation pipeline's layout vocabulary
  (`src/common/state.py::LayoutBlockKind`) has no dedicated `THEOREM`/`LEMMA`/`PROOF` block kind (only
  `DEFINITION`); generating documents against this benchmark's theorem/proof-heavy items may require extending
  that enum in a follow-up change -- out of scope for the dataset-construction pipeline itself.
