"""
PHASES OF THIS FILE (BY HRISHABH :)

1. Foundation (Import & Setup)
2. Data Classes (Containers)
3. Helper Functions (Building Blocks)
4. Core Analysis (Main Logic)
5. Integration (Final Touch)

"""



# PHASE 1

from __future__ import annotations # To use type hints

import re # "Regular Expression" : To find Var. names in expressions

from dataclasses import dataclass, field
# Creates container classes automatically
# To customize how dataclass fields behave

from typing import Dict, List, Optional # To describe these types

from .parser import ArrayAccess, KernelInfo, LoopNest, LoopVar
# Imports data classes from parser.py




# PHASE 2

    # CLASS 1 : Info about a single array access
@dataclass
class AccessQuality:
    access : ArrayAccess # Original Data [Name,indices]
    quality : str # "good" "poor" "unknown"
    reason : str 

    # CLASS 2 : Complete Analysis of one loop nest
@dataclass
class NestAnalysis:
    nest : LoopNest # Ref. to original loop structure
    access_qualities : List[AccessQuality] # Rate each array acces individually
    overall_quality : str # "good" "poor" "mixed" "unknown"
    recommend_interchange : bool
    recommend_tiling : bool
    recommendation_notes : List[str]

    # CLASS 3 : Complete Analysis of all loops in a function
@dataclass
class AnalysisReport:
    kernel_info : KernelInfo # Function name,file,etc
    nest_analyses : List[NestAnalysis]

    def summary(self) -> str:
        lines = []

        # Header : Function name and file
        lines.append(f"Kernel function : {self.kernel_info.func_name}")
        lines.append(f"Source file     : {self.kernel_info.filename}")

        # Show analysis for each loop nest
        for nest_num, nest_analysis in enumerate(self.nest_analyses, 1):
            lines.append(f"\nLoop Nest #{nest_num}:")

            # Structure of the Loop
            loop_vars = nest_analysis.nest.loops
            loop_structure = "->".join(
                f"for {loop.name} : [{loop.start}, {loop.end})"
                for loop in loop_vars # Generator Expression
            )
            lines.append(f" Structure : {loop_structure}")

            # Overall locality grade
            lines.append(f" Locality : {nest_analysis.overall_quality.upper()}")

            # Each array access and its rating
            lines.append(" Accesses :")
            for access_quality in nest_analysis.access_qualities:
                access_type = "write" if access_quality.access.is_write else "read "
                index_string = "[" + "][".join(access_quality.access.indices) + "]"
                # Format : A[i][j] (read) - GOOD: Explaination
                lines.append(
                    f"    {access_quality.access.array_name}{index_string} "
                    f"({access_type}) — "
                    f"{access_quality.quality.upper()}: {access_quality.reason}"
                )

            # Show Recommendations
            if nest_analysis.recommendation_notes:
                lines.append("  Recommendations:")
                for note in nest_analysis.recommendation_notes:
                    lines.append(f"    → {note}")

        # Join all lines with \n and return as one string
        return "\n".join(lines)



# PHASE 3

# Function to find if a variable exits standalone in a regular expression
def _contains_var(expr: str, var: str) -> bool:
    return bool(re.search(rf"\b{re.escape(var)}\b", expr))

# THE MOST IMPORTANT FUNCTION [CORE LOGIC]
def _classify_access(access: ArrayAccess, inner_var: str) -> AccessQuality:
    
    # Don't know the indices
    if not access.indices:
        return AccessQuality(
            access=access,
            quality="unknown",
            reason="no index information"
        )

    last_idx = access.indices[-1]
    dims = len(access.indices)
    
    #If Inner variable is the lst index
    if _contains_var(last_idx, inner_var):
        reason = (
            f"inner var '{inner_var}' varies last index → "
            f"sequential {'writes' if access.is_write else 'reads'}"
        )
        return AccessQuality(
            access=access,
            quality="good",
            reason=reason
        )
    
    # If array is 1-D [Always good]
    if dims == 1:
        if _contains_var(access.indices[0], inner_var):
            reason = f"1-D sequential access on '{inner_var}'"
        else:
            reason = f"inner var '{inner_var}' absent from subscript (scalar reuse)"
        
        return AccessQuality(
            access=access,
            quality="good",
            reason=reason
        )
    
    # If Inner variable is non-last
    earlier_indices = access.indices[:-1]
    
    if any(_contains_var(idx, inner_var) for idx in earlier_indices):
        stride = "row" if dims == 2 else "plane/row"
        reason = (
            f"inner var '{inner_var}' varies {stride} index → "
            f"strided {'writes' if access.is_write else 'reads'}"
        )
        return AccessQuality(
            access=access,
            quality="poor",
            reason=reason
        )
    
    # Inner Variable doesn't appear anywhere
    reason = f"array is invariant w.r.t. inner var '{inner_var}' (reuse)"
    return AccessQuality(
        access=access,
        quality="good",
        reason=reason
    )






# PHASE 4


# Function to analyze a single nest
def analyse_nest(nest: LoopNest) -> NestAnalysis:
    
    # If no loops exist
    if not nest.loops:
        return NestAnalysis(
            nest=nest,
            access_qualities=[],
            overall_quality="unknown",
            recommend_interchange=False,
            recommend_tiling=False,
            recommendation_notes=["Empty loop nest"],
        )
    
    # Innermost Loop-Variable
    inner_var = nest.loops[-1].name
    
    # Classify each Array-Access
    access_qualities: List[AccessQuality] = []
    for acc in nest.accesses:
        aq = _classify_access(acc, inner_var)
        access_qualities.append(aq)
    
    # Calculate Overall Quality
    grades = {aq.quality for aq in access_qualities}
    
    if not grades or grades == {"unknown"}:
        overall = "unknown"
    elif "poor" not in grades:
        overall = "good"
    elif "good" not in grades:
        overall = "poor"
    else:
        overall = "mixed"
    
    notes: List[str] = []
    recommend_interchange = False
    recommend_tiling = False
    
    depth = len(nest.loops) # Nesting Weight
    
    # If Overall Quality is good
    if overall == "good":
        notes.append("Access pattern is already cache-friendly.")
        # If nested loop
        if depth >= 2:
            recommend_tiling = True
            notes.append(
                "Loop tiling can improve register reuse even for "
                "cache-friendly patterns."
            )
    
    # If Overall Quality is Poor or Mixed
    elif overall in ("poor", "mixed"):
        if depth >= 2:
            # Try to see if loop interchange would help
            outer_var = nest.loops[0].name  # Outermost Loop Variable
            
            interchange_grades: List[str] = []
            for acc in nest.accesses:
                aq2 = _classify_access(acc, outer_var)
                interchange_grades.append(aq2.quality)

            # Calculate interchange "good" and current "good"
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


# Analyze all loop nests in a single kernel function
def analyse(kernel_info: KernelInfo) -> AnalysisReport:
    
    nest_analyses = [analyse_nest(n) for n in kernel_info.loop_nests]
    
    return AnalysisReport(
        kernel_info=kernel_info,
        nest_analyses=nest_analyses,
    )


# Analyze all kernel functions in a C File
def analyse_file(filepath: str, func_name: Optional[str] = None) -> List[AnalysisReport]:
    # Import the parser
    from .parser import parse_kernel
    
    # Parse the file to get all kernel functions
    all_kernels = parse_kernel(filepath)
    
    # Analyze each kernel
    reports = []
    for ki in all_kernels:
        # if func_name was specified and doesn't match, then skip
        if func_name and ki.func_name != func_name:
            continue
        
        # Skip main function
        if ki.func_name == "main":
            continue
        
        # Analyze this kernel
        reports.append(analyse(ki))
    
    return reports
