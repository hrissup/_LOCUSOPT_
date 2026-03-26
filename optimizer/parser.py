"""
optimizer/parser.py

Parse a restricted subset of C numerical kernels using pycparser.

1. Preprocess with ``gcc -E -std=c99`` (preserves linemarkers so that
   pycparser coords map back to the *original* source lines).
2. Parse the preprocessed output into a pycparser AST.
3. Walk the AST to extract :class:`KernelInfo` objects (one per function
   definition that contains nested for-loops).

Supported patterns

- Canonical ``for (var = start; var < end; var++)`` loops
- 1-D and 2-D array references whose subscripts are simple expressions
  of loop induction variables
- Functions with at most one level of nested loop nesting depth = 2 or 3
"""

from __future__ import annotations

import copy
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pycparser
from pycparser import c_ast, c_generator

_gen = c_generator.CGenerator()


@dataclass
class ArrayAccess:
    """A single array read or write inside a loop body."""
    array_name: str
    indices: List[str]      # e.g. ["i", "j"] for A[i][j]
    is_write: bool
    line: int = 0


@dataclass
class LoopVar:
    """Induction variable of a single ``for`` loop."""
    name: str
    start: str              # symbolic lower bound (inclusive), e.g. "0"
    end: str                # symbolic upper bound (exclusive), e.g. "512"
    step: int = 1           # assumed +1 in supported patterns


@dataclass
class LoopNest:
    """A sequence of nested for-loops together with all array accesses."""
    loops: List[LoopVar]            # outermost first
    accesses: List[ArrayAccess]
    body_text: str = ""             # CGenerator text of innermost body
    start_line: int = 0             # 1-based line in original file
    end_line: int = 0


@dataclass
class KernelInfo:
    """Parsed representation of a single kernel function."""
    filename: str
    func_name: str
    loop_nests: List[LoopNest]
    ast_node: object = field(default=None, repr=False)  # pycparser FuncDef


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _expr_str(node) -> str:
    """Convert a pycparser expression node to a string."""
    if node is None:
        return ""
    return _gen.visit(node)


def _extract_array_access(node: c_ast.ArrayRef, is_write: bool) -> ArrayAccess:
    """
    Recursively unwrap ``ArrayRef`` to get name + ordered index list.

    For ``A[i][j]`` the pycparser tree is::

        ArrayRef(name=ArrayRef(name=ID('A'), subscript=ID('i')),
                 subscript=ID('j'))

    so indices come out as ``["i", "j"]``.
    """
    indices: List[str] = []
    current = node
    while isinstance(current, c_ast.ArrayRef):
        indices.insert(0, _expr_str(current.subscript))
        current = current.name
    name = _expr_str(current)
    line = node.coord.line if node.coord else 0
    return ArrayAccess(array_name=name, indices=indices,
                       is_write=is_write, line=line)


def _parse_for_var(for_node: c_ast.For) -> Optional[LoopVar]:
    """
    Extract :class:`LoopVar` from a ``for`` statement.

    Supports both::

        for (i = 0; i < N; i++)           # Assignment init
        for (int i = 0; i < N; i++)       # DeclList init (C99)

    Returns ``None`` if the loop does not match the canonical pattern.
    """
    # --- init ---
    init = for_node.init
    var_name: Optional[str] = None
    start: Optional[str] = None

    if isinstance(init, c_ast.Assignment) and isinstance(init.lvalue, c_ast.ID):
        var_name = init.lvalue.name
        start = _expr_str(init.rvalue)
    elif isinstance(init, c_ast.DeclList) and len(init.decls) == 1:
        decl = init.decls[0]
        var_name = decl.name
        start = _expr_str(decl.init) if decl.init else "0"
    else:
        return None

    # --- cond: var < end ---
    cond = for_node.cond
    if not isinstance(cond, c_ast.BinaryOp):
        return None
    if cond.op not in ("<", "<="):
        return None
    end_expr = _expr_str(cond.right)
    # Adjust for <= (make it exclusive)
    if cond.op == "<=":
        end_expr = f"({end_expr}) + 1"

    # --- next: var++ or ++var or var += 1 ---
    nxt = for_node.next
    step = 1
    if isinstance(nxt, c_ast.UnaryOp) and nxt.op in ("p++", "++"):
        pass  # step = 1
    elif isinstance(nxt, c_ast.Assignment) and nxt.op == "+=":
        rhs = _expr_str(nxt.rvalue)
        try:
            step = int(rhs)
        except ValueError:
            return None
    else:
        return None

    if var_name is None or start is None:
        return None

    return LoopVar(name=var_name, start=start, end=end_expr, step=step)


# ---------------------------------------------------------------------------
# Loop-nest extraction visitor
# ---------------------------------------------------------------------------

class _LoopNestExtractor(c_ast.NodeVisitor):
    """
    Collect all top-level for-loop nests inside a function.

    A "nest" is a chain of directly nested for-loops (no other statements
    between levels).  We stop when the innermost for body contains non-for
    statements.
    """

    def __init__(self, func_name: str):
        self.func_name = func_name
        self.nests: List[LoopNest] = []
        self._in_func = False

    # ------------------------------------------------------------------
    def visit_FuncDef(self, node: c_ast.FuncDef):
        if node.decl.name == self.func_name:
            self._in_func = True
            self.generic_visit(node)
            self._in_func = False

    # ------------------------------------------------------------------
    def visit_For(self, node: c_ast.For):
        if not self._in_func:
            return
        # Only process top-level for-loops in this function (not nested)
        # We'll handle nesting below
        nest = self._collect_nest(node)
        if nest is not None:
            self.nests.append(nest)
        # Do NOT recurse further; _collect_nest handles children

    # ------------------------------------------------------------------
    def _collect_nest(self, for_node: c_ast.For) -> Optional[LoopNest]:
        """
        Walk nested for-loops and collect the full nest.
        Stop at the first level whose body contains non-for statements.
        Return ``None`` if the outermost level is not parseable.
        """
        loops: List[LoopVar] = []
        accesses: List[ArrayAccess] = []

        current = for_node
        while isinstance(current, c_ast.For):
            lv = _parse_for_var(current)
            if lv is None:
                break
            loops.append(lv)

            # Unwrap Compound wrapper
            body = current.stmt
            if isinstance(body, c_ast.Compound) and body.block_items:
                inner = body.block_items
            elif isinstance(body, c_ast.Compound):
                inner = []
            else:
                inner = [body]

            # Check if body is a single nested for-loop
            if len(inner) == 1 and isinstance(inner[0], c_ast.For):
                current = inner[0]
            else:
                # This is the innermost body — collect accesses
                for stmt in inner:
                    _collect_accesses(stmt, accesses)
                body_text = _gen.visit(current.stmt) if current.stmt else ""
                start_line = for_node.coord.line if for_node.coord else 0
                return LoopNest(
                    loops=loops,
                    accesses=accesses,
                    body_text=body_text,
                    start_line=start_line,
                )

        if loops:
            # Degenerate: nest only, no body found
            body_text = _gen.visit(for_node.stmt) if for_node.stmt else ""
            return LoopNest(loops=loops, accesses=accesses,
                            body_text=body_text,
                            start_line=for_node.coord.line if for_node.coord else 0)
        return None


def _collect_accesses(node, accesses: List[ArrayAccess]):
    """Recursively collect all ArrayRef nodes as reads or writes."""
    if node is None:
        return

    if isinstance(node, c_ast.Assignment):
        # LHS is a write; we must collect accesses in LHS carefully
        _mark_write(node.lvalue, accesses)
        _collect_accesses(node.rvalue, accesses)
    elif isinstance(node, c_ast.ArrayRef):
        # In a non-assignment context this is a read
        acc = _extract_array_access(node, is_write=False)
        # Avoid duplicating nested refs already captured by _mark_write
        accesses.append(acc)
    elif isinstance(node, c_ast.Compound):
        for item in (node.block_items or []):
            _collect_accesses(item, accesses)
    else:
        for _, child in node.children():
            _collect_accesses(child, accesses)


def _mark_write(node, accesses: List[ArrayAccess]):
    """Mark the LHS of an assignment; outermost ArrayRef is the write target."""
    if isinstance(node, c_ast.ArrayRef):
        acc = _extract_array_access(node, is_write=True)
        accesses.append(acc)
        # Inner indices of the LHS are reads — don't recurse into name
    else:
        _collect_accesses(node, accesses)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_kernel(filepath: str) -> List[KernelInfo]:
    """
    Parse *filepath* (a C source file) and return a list of
    :class:`KernelInfo` objects, one per function definition that
    contains at least one analysable nested for-loop.

    Uses ``gcc -E`` to preprocess, preserving linemarkers so that
    AST coords map to the *original* file.
    """
    filepath = os.path.abspath(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")

    # Parse via pycparser (use_cpp=True runs gcc -E internally)
    try:
        ast = pycparser.parse_file(
            filepath,
            use_cpp=True,
            cpp_path="gcc",
            cpp_args=["-E", "-std=c99",
                      # Suppress GCC-specific attributes that confuse pycparser
                      "-D__attribute__(x)=",
                      "-D__extension__=",
                      "-D__restrict=",
                      "-D__inline="],
        )
    except pycparser.plyparser.ParseError as exc:
        raise SyntaxError(
            f"pycparser failed to parse '{filepath}':\n{exc}"
        ) from exc

    results: List[KernelInfo] = []
    for ext_node in ast.ext:
        if not isinstance(ext_node, c_ast.FuncDef):
            continue
        func_name = ext_node.decl.name
        extractor = _LoopNestExtractor(func_name)
        extractor.visit(ext_node)
        if extractor.nests:
            results.append(KernelInfo(
                filename=filepath,
                func_name=func_name,
                loop_nests=extractor.nests,
                ast_node=ext_node,
            ))

    return results


def parse_kernel_function(filepath: str, func_name: str) -> Optional[KernelInfo]:
    """Return the :class:`KernelInfo` for a specific function, or None."""
    for ki in parse_kernel(filepath):
        if ki.func_name == func_name:
            return ki
    return None


def get_ast(filepath: str):
    """Return the raw pycparser AST for *filepath*."""
    filepath = os.path.abspath(filepath)
    return pycparser.parse_file(
        filepath,
        use_cpp=True,
        cpp_path="gcc",
        cpp_args=["-E", "-std=c99",
                  "-D__attribute__(x)=",
                  "-D__extension__=",
                  "-D__restrict=",
                  "-D__inline="],
    )
