

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
    array_name: str
    indices: List[str]      
    is_write: bool
    line: int = 0


@dataclass
class LoopVar:
    name: str
    start: str            
    end: str              
    step: int = 1      


@dataclass
class LoopNest:
    loops: List[LoopVar]         
    accesses: List[ArrayAccess]
    body_text: str = ""          
    start_line: int = 0             
    end_line: int = 0


@dataclass
class KernelInfo:
    filename: str
    func_name: str
    loop_nests: List[LoopNest]
    ast_node: object = field(default=None, repr=False)



def _expr_str(node) -> str:
    if node is None:
        return ""
    return _gen.visit(node)


def _extract_array_access(node: c_ast.ArrayRef, is_write: bool) -> ArrayAccess:
 
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

    cond = for_node.cond
    if not isinstance(cond, c_ast.BinaryOp):
        return None
    if cond.op not in ("<", "<="):
        return None
    end_expr = _expr_str(cond.right)
    if cond.op == "<=":
        end_expr = f"({end_expr}) + 1"

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



class _LoopNestExtractor(c_ast.NodeVisitor):
    
    def __init__(self, func_name: str):
        self.func_name = func_name
        self.nests: List[LoopNest] = []
        self._in_func = False

    def visit_FuncDef(self, node: c_ast.FuncDef):
        if node.decl.name == self.func_name:
            self._in_func = True
            self.generic_visit(node)
            self._in_func = False

    def visit_For(self, node: c_ast.For):
        if not self._in_func:
            return

        nest = self._collect_nest(node)
        if nest is not None:
            self.nests.append(nest)

    def _collect_nest(self, for_node: c_ast.For) -> Optional[LoopNest]:
      
        loops: List[LoopVar] = []
        accesses: List[ArrayAccess] = []

        current = for_node
        while isinstance(current, c_ast.For):
            lv = _parse_for_var(current)
            if lv is None:
                break
            loops.append(lv)

            body = current.stmt
            if isinstance(body, c_ast.Compound) and body.block_items:
                inner = body.block_items
            elif isinstance(body, c_ast.Compound):
                inner = []
            else:
                inner = [body]

            if len(inner) == 1 and isinstance(inner[0], c_ast.For):
                current = inner[0]
            else:
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
            body_text = _gen.visit(for_node.stmt) if for_node.stmt else ""
            return LoopNest(loops=loops, accesses=accesses,
                            body_text=body_text,
                            start_line=for_node.coord.line if for_node.coord else 0)
        return None


def _collect_accesses(node, accesses: List[ArrayAccess]):
    if node is None:
        return

    if isinstance(node, c_ast.Assignment):
        _mark_write(node.lvalue, accesses)
        _collect_accesses(node.rvalue, accesses)
    elif isinstance(node, c_ast.ArrayRef):
        acc = _extract_array_access(node, is_write=False)
        accesses.append(acc)
    elif isinstance(node, c_ast.Compound):
        for item in (node.block_items or []):
            _collect_accesses(item, accesses)
    else:
        for _, child in node.children():
            _collect_accesses(child, accesses)


def _mark_write(node, accesses: List[ArrayAccess]):
    if isinstance(node, c_ast.ArrayRef):
        acc = _extract_array_access(node, is_write=True)
        accesses.append(acc)
    else:
        _collect_accesses(node, accesses)



def parse_kernel(filepath: str) -> List[KernelInfo]:
   
    filepath = os.path.abspath(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")

    try:
        ast = pycparser.parse_file(
            filepath,
            use_cpp=True,
            cpp_path="gcc",
            cpp_args=["-E", "-std=c99",
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
    for ki in parse_kernel(filepath):
        if ki.func_name == func_name:
            return ki
    return None


def get_ast(filepath: str):
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
