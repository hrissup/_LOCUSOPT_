from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .parser import ArrayAccess, LoopNest, LoopVar


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DependenceResult:
    """Outcome of dependence analysis for a loop nest."""
    safe_interchange: bool
    safe_tiling: bool
    reason: str                     # human-readable explanation
    details: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOOP_CARRIED_PATTERN = re.compile(
    r"\b(\w+)\s*[+\-]\s*\d+",     # e.g.  i - 1,  j + 2
)


def _is_stencil_index(expr: str, loop_vars: Set[str]) -> bool:
    """
    Return True if *expr* looks like a stencil (loop-carried) index such
    as ``i-1``, ``j+1``.  These are safe to tile if the same array is only
    *read* (not written) with such an expression.
    """
    for m in _LOOP_CARRIED_PATTERN.finditer(expr):
        if m.group(1) in loop_vars:
            return True
    return False


def _is_affine(expr: str, loop_vars: Set[str]) -> bool:
    """
    Very lightweight check: an index is considered affine if it contains
    only identifiers, integers, ``+``, ``-``, ``*``, spaces, and
    parentheses.  Non-affine constructs (function calls, ``/``, ``%``,
    array subscripts inside subscripts, etc.) are flagged as non-affine.
    """
    non_affine = re.search(r"[/%\[\]]|[a-zA-Z_]\w*\s*\(", expr)
    return non_affine is None


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def check_dependence(nest: LoopNest) -> DependenceResult:
    """
    Perform conservative dependence analysis on *nest*.

    Returns a :class:`DependenceResult` indicating whether loop
    interchange and/or loop tiling are safe.
    """
    if len(nest.loops) < 2:
        return DependenceResult(
            safe_interchange=False,
            safe_tiling=False,
            reason="Need at least 2 loop levels for interchange/tiling.",
        )

    loop_vars: Set[str] = {lv.name for lv in nest.loops}
    details: List[str] = []

    # ------------------------------------------------------------------ #
    # 1. Check for non-affine indices
    # ------------------------------------------------------------------ #
    non_affine_found = False
    for acc in nest.accesses:
        for idx in acc.indices:
            if not _is_affine(idx, loop_vars):
                details.append(
                    f"Non-affine index '{idx}' in "
                    f"{'write' if acc.is_write else 'read'} "
                    f"of '{acc.array_name}'."
                )
                non_affine_found = True

    if non_affine_found:
        return DependenceResult(
            safe_interchange=False,
            safe_tiling=False,
            reason="Non-affine index expression detected — skipping transformation.",
            details=details,
        )

    # ------------------------------------------------------------------ #
    # 2. Separate reads and writes per array
    # ------------------------------------------------------------------ #
    writes: Dict[str, List[ArrayAccess]] = {}
    reads:  Dict[str, List[ArrayAccess]] = {}
    for acc in nest.accesses:
        d = writes if acc.is_write else reads
        d.setdefault(acc.array_name, []).append(acc)

    # ------------------------------------------------------------------ #
    # 3. Check read-after-write (RAW) and write-after-read (WAR) deps
    #    across different index tuples
    # ------------------------------------------------------------------ #
    has_loop_carried_dep = False
    stencil_read_only: Set[str] = set()  # arrays read with stencil but not written

    for arr, write_accesses in writes.items():
        write_indices = [tuple(wa.indices) for wa in write_accesses]

        if arr in reads:
            # Array both read and written
            for ra in reads[arr]:
                r_idx = tuple(ra.indices)
                # If any read index has a stencil offset (e.g. A[i-1][j])
                # and the array is also written, there is a loop-carried dep
                for idx_expr in ra.indices:
                    if _is_stencil_index(idx_expr, loop_vars):
                        details.append(
                            f"Loop-carried dependence: '{arr}' is written and "
                            f"read with stencil index '{idx_expr}'."
                        )
                        has_loop_carried_dep = True
                # If read and write use the *same* tuple → point-wise, safe
                if r_idx not in write_indices:
                    # Different indices — check if this could be a dep
                    for w_idx in write_indices:
                        if r_idx != w_idx:
                            details.append(
                                f"Potential dependence: '{arr}' written at "
                                f"{list(w_idx)} and read at {list(r_idx)}."
                            )
                            # Only mark as dep if indices differ by a stencil
                            if any(_is_stencil_index(i, loop_vars)
                                   for i in list(r_idx) + list(w_idx)):
                                has_loop_carried_dep = True
        else:
            # Written but not read — check write indices for stencil
            for wa in write_accesses:
                for idx_expr in wa.indices:
                    if _is_stencil_index(idx_expr, loop_vars):
                        details.append(
                            f"Write with stencil index to '{arr}' at '{idx_expr}'."
                        )

    for arr, read_accesses in reads.items():
        if arr not in writes:
            for ra in read_accesses:
                for idx in ra.indices:
                    if _is_stencil_index(idx, loop_vars):
                        stencil_read_only.add(arr)

    if stencil_read_only:
        details.append(
            f"Read-only stencil accesses (safe): {sorted(stencil_read_only)}"
        )

    # ------------------------------------------------------------------ #
    # 4. Decide
    # ------------------------------------------------------------------ #
    if has_loop_carried_dep:
        return DependenceResult(
            safe_interchange=False,
            safe_tiling=False,
            reason="Loop-carried dependence detected — transformation not safe.",
            details=details,
        )

    depth = len(nest.loops)
    safe_interchange = depth == 2 and not has_loop_carried_dep
    safe_tiling = not has_loop_carried_dep

    reason = "No loop-carried dependencies detected — transformations are safe."
    if stencil_read_only:
        reason += (
            f" Read-only stencil on {sorted(stencil_read_only)} — "
            "tiling must preserve tile boundaries for correctness."
        )

    return DependenceResult(
        safe_interchange=safe_interchange,
        safe_tiling=safe_tiling,
        reason=reason,
        details=details,
    )
