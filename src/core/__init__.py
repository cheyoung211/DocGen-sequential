"""Cross-cutting concepts for the dependency- and budget-aware controller
revision (Notion "Revision 2026.10.15").

Everything here is additive to the existing pipeline (``src/agents``,
``src/common``, ``src/evaluation``) -- nothing in this package is imported
by, or changes the behavior of, ``scripts/run_full_pipeline.py`` yet. See
``docs/stage1_experiment_design.md`` for the experiment design this package
implements interfaces for, and each module's own docstring for how it
relates to the pipeline code it wraps.
"""
