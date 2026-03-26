"""
optimizer/analyzer.py

Cache-locality analysis for loop nests extracted by the parser.

Theory
------
C/C++ stores multi-dimensional arrays in *row-major* (last-index-fastest)
order.  For a 2-D array ``A[rows][cols]``:

    A[i][j]  is stored at  base + i*cols + j

When the *inner-most* loop variable varies the **last** (rightmost) index,
successive iterations access consecutive memory addresses → cache-friendly.
When the inner-most loop variable varies an earlier index, accesses are
*strided* (typically by ``sizeof(element) * cols``) → cache-unfriendly.

Locality grades
---------------
``"good"``    All accesses have their last index dominated by the inner var.
``"poor"``    All accesses violate the row-major rule.
``"mixed"``   Some accesses are good, some are poor.
``"unknown"`` Cannot classify (complex index expressions).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .parser import ArrayAccess, KernelInfo, LoopNest, LoopVar


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AccessQuality:
    """Cache locality quality of a single array access."""
    access: ArrayAccess
    quality: str            # "good" | "poor" | "unknown"
    reason: str


@dataclass
class NestAnalysis:
    """Analysis result for a single loop nest."""
    nest: LoopNest
    access_qualities: List[AccessQuality]
    overall_quality: str    # "good" | "poor" | "mixed" | "unknown"
    recommend_interchange: bool
    recommend_tiling: bool
    recommendation_notes: List[str]


@dataclass
class AnalysisReport:
    """Full analysis report for a kernel file."""
    kernel_info: KernelInfo
    nest_analyses: List[NestAnalysis]

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = []
        lines.append(f"Kernel function : {self.kernel_info.func_name}")
        lines.append(f"Source file     : {self.kernel_info.filename}")
        for idx, na in enumerate(self.nest_analyses, 1):
            lines.append(f"\nLoop Nest #{idx}:")
            lvs = na.nest.loops
            struct = " → ".join(
                f"for {lv.name} ∈ [{lv.start}, {lv.end})" for lv in lvs
            )
            lines.append(f"  Structure : {struct}")
            lines.append(f"  Locality  : {na.overall_quality.upper()}")
            lines.append("  Accesses  :")
            for aq in na.access_qualities:
                mode = "write" if aq.access.is_write else "read "
                idx_str = "[" + "][".join(aq.access.indices) + "]"
                lines.append(
                    f"    {aq.access.array_name}{idx_str} ({mode}) — "
                    f"{aq.quality.upper()}: {aq.reason}"
                )
            if na.recommendation_notes:
                lines.append("  Recommendations:")
                for note in na.recommendation_notes:
                    lines.append(f"    → {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contains_var(expr: str, var: str) -> bool:
    """Return True if *expr* contains *var* as a standalone identifier."""
    return bool(re.search(rf"\b{re.escape(var)}\b", expr))


def _classify_access(access: ArrayAccess, inner_var: str) -> AccessQuality:
    """
    Classify a single array access against the inner-most loop variable.

    For ``A[i][j]`` (indices = ["i", "j"]) with inner var ``j``:
      - Last index is "j" → contains inner var → **good** (row-major).
    For ``B[j][i]`` (indices = ["j", "i"]) with inner var ``j``:
      - Last index is "i" → does NOT contain inner var → **poor**.
    """
    if not access.indices:
        return AccessQuality(access, "unknown", "no index information")

    last_idx = access.indices[-1]
    dims = len(access.indices)

    if _contains_var(last_idx, inner_var):
        reason = (
            f"inner var '{inner_var}' varies last index → sequential "
            f"{'writes' if access.is_write else 'reads'}"
        )
        return AccessQuality(access, "good", reason)

    if dims == 1:
        # 1-D array with inner var not in subscript: static in inner loop
        if _contains_var(access.indices[0], inner_var):
            reason = f"1-D sequential access on '{inner_var}'"
            return AccessQuality(access, "good", reason)
        reason = f"inner var '{inner_var}' absent from subscript (scalar reuse)"
        return AccessQuality(access, "good", reason)  # invariant is fine

    # Check if inner_var appears in ANY index (non-last): that is the bad case
    earlier_indices = access.indices[:-1]
    if any(_contains_var(idx, inner_var) for idx in earlier_indices):
        stride = "row" if dims == 2 else "plane/row"
        reason = (
            f"inner var '{inner_var}' varies {stride} index → strided "
            f"{'writes' if access.is_write else 'reads'}"
        )
        return AccessQuality(access, "poor", reason)

    # Inner var not present anywhere → invariant in inner loop (reuse)
    reason = f"array is invariant w.r.t. inner var '{inner_var}' (reuse)"
    return AccessQuality(access, "good", reason)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse_nest(nest: LoopNest) -> NestAnalysis:
    """
    Analyse the locality of a single :class:`LoopNest`.

    Returns a :class:`NestAnalysis`.
    """
    if not nest.loops:
        return NestAnalysis(
            nest=nest,
            access_qualities=[],
            overall_quality="unknown",
            recommend_interchange=False,
            recommend_tiling=False,
            recommendation_notes=["Empty loop nest"],
        )

    inner_var = nest.loops[-1].name
    access_qualities: List[AccessQuality] = []

    for acc in nest.accesses:
        aq = _classify_access(acc, inner_var)
        access_qualities.append(aq)

    # Overall quality
    grades = {aq.quality for aq in access_qualities}
    if not grades or grades == {"unknown"}:
        overall = "unknown"
    elif "poor" not in grades:
        overall = "good"
    elif "good" not in grades:
        overall = "poor"
    else:
        overall = "mixed"

    # Recommendations
    notes: List[str] = []
    recommend_interchange = False
    recommend_tiling = False

    depth = len(nest.loops)

    if overall == "good":
        notes.append("Access pattern is already cache-friendly.")
        if depth >= 2:
            recommend_tiling = True
            notes.append(
                "Loop tiling can improve register/TLB reuse even for "
                "cache-friendly patterns."
            )
    elif overall in ("poor", "mixed"):
        if depth >= 2:
            # Evaluate whether interchange would help
            outer_var = nest.loops[0].name if depth >= 2 else None
            if outer_var is not None:
                # Simulate interchange: outer_var becomes new inner var
                interchange_grades: List[str] = []
                for acc in nest.accesses:
                    aq2 = _classify_access(acc, outer_var)
                    interchange_grades.append(aq2.quality)
                interchange_good = interchange_grades.count("good")
                current_good = sum(1 for aq in access_qualities if aq.quality == "good")

                if interchange_good > current_good:
                    recommend_interchange = True
                    notes.append(
                        f"Loop interchange (swap '{nest.loops[0].name}' ↔ "
                        f"'{nest.loops[-1].name}') improves locality for "
                        f"{interchange_good} / {len(nest.accesses)} accesses."
                    )
                else:
                    notes.append(
                        "Loop interchange does not improve overall locality "
                        "(mixed-access pattern)."
                    )

                # Tiling is almost always beneficial for mixed/poor 2-D nests
                recommend_tiling = True
                notes.append(
                    "Loop tiling (blocking) is recommended — reduces cache "
                    "miss rate by operating on tile-sized sub-matrices that "
                    "fit in L1/L2 cache."
                )

    if depth >= 3:
        recommend_tiling = True
        notes.append(
            "3-level nest detected — tiling on all three loops is "
            "beneficial (e.g. matrix multiply blocking)."
        )

    return NestAnalysis(
        nest=nest,
        access_qualities=access_qualities,
        overall_quality=overall,
        recommend_interchange=recommend_interchange,
        recommend_tiling=recommend_tiling,
        recommendation_notes=notes,
    )


def analyse(kernel_info: KernelInfo) -> AnalysisReport:
    """
    Analyse all loop nests in *kernel_info*.

    Only analyses nests belonging to the primary kernel function
    (i.e. not ``main``).
    """
    nest_analyses = [analyse_nest(n) for n in kernel_info.loop_nests]
    return AnalysisReport(
        kernel_info=kernel_info,
        nest_analyses=nest_analyses,
    )


def analyse_file(filepath: str, func_name: Optional[str] = None) -> List[AnalysisReport]:
    """
    Parse *filepath* and analyse all kernel functions (or just *func_name*).

    Returns a list of :class:`AnalysisReport` objects.
    """
    from .parser import parse_kernel

    all_kernels = parse_kernel(filepath)
    reports = []
    for ki in all_kernels:
        if func_name and ki.func_name != func_name:
            continue
        if ki.func_name == "main":
            continue
        reports.append(analyse(ki))
    return reports
