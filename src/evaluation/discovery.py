"""Shared batch-directory discovery, lifted from the 3 existing offline
evaluators (``src/evaluation/{latex_robustness,long_form_quality,
multimodal_alignment}/``), which each independently redefine an identical
copy. Those 3 files are left untouched -- this only gives the new
``aggregate.py`` one copy to import instead of a 4th duplicate.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple


def discover_sample_dirs(base_dir: Path) -> List[Path]:
    """Directories with evidence that generation was attempted (a plan.json or
    meta.json was written), regardless of whether it produced a compilable
    main.tex. A directory with neither marker never started generating and is
    excluded entirely rather than reported as a failure.
    """
    marker_files = list(base_dir.glob("**/plan.json")) + list(base_dir.glob("**/meta.json"))
    return sorted({f.parent for f in marker_files})


def split_factor_sample(relative_dir: Path) -> Tuple[Optional[str], str]:
    """Recover (factor, sample_id) from a sample directory relative to the batch dir.

    Supports both the factor/sample_id layout and the flatter sample_id-only
    layout (no stress-factor grouping) -- the latter is what
    ``run_batch()``'s ``request_id = f"{group}/sample_{index:03d}"`` produces
    for new-schema benchmark items, where ``group`` is the sample_id.
    """
    path_parts = relative_dir.parts
    if len(path_parts) >= 2:
        return path_parts[0], path_parts[1]
    if len(path_parts) == 1:
        return None, path_parts[0]
    return None, "unknown"
