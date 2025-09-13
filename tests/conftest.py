
import types
import ast
import nbformat
import builtins
import sys
import re 
from pathlib import Path
import importlib

NOTEBOOK_PATH = Path(r"C:\Users\Cliente.pc\Desktop\arti\Transformer_Nahuatl_Espanol_FromScratch.ipynb")


def _extract_defs_and_imports_from_cell(src: str) -> str:
    """
    Parse a code cell and keep only:
      - import and from-import statements
      - function/class definitions
      - simple assignments that define constants (UPPER_CASE) and hyperparams
    Everything else (training loops, heavy execution) is stripped.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""

    keep_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            keep_nodes.append(node)
        elif isinstance(node, ast.Assign):
            # Keep only simple constant assignments (literals, numbers, strings, tuples of literals)
            def is_const(n):
                return isinstance(n, (ast.Constant, ast.Num, ast.Str)) or (
                    isinstance(n, (ast.Tuple, ast.List)) and all(isinstance(elt, (ast.Constant, ast.Num, ast.Str)) for elt in n.elts)
                )
            if all(isinstance(t, ast.Name) for t in node.targets) and is_const(node.value):
                keep_nodes.append(node)
        elif isinstance(node, ast.AnnAssign):
            # Keep annotated assignments with constants
            val = node.value
            if val is not None and isinstance(val, (ast.Constant, ast.Num, ast.Str)):
                keep_nodes.append(node)

    # Reconstruct code from kept nodes
    module = ast.Module(body=keep_nodes, type_ignores=[])
    try:
        return ast.unparse(module)  # Python 3.9+
    except Exception:
        # Fallback: do nothing if unparse fails
        return ""

def load_notebook_namespace():
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Notebook not found at {NOTEBOOK_PATH}")
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_chunks = []
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            code_chunks.append(_extract_defs_and_imports_from_cell(cell.get("source", "")))

    filtered_src = "\\n\\n".join([c for c in code_chunks if c.strip()])

    # --- SANITIZADOR PARA EVITAR SyntaxError POR "\" DE CONTINUACIÓN ---
    # 1) Quita continuaciones de línea con backslash al final
    filtered_src = re.sub(r"\\\r?\n", "\n", filtered_src)
    # 2) Quita backslashes sueltos seguidos de espacios (a veces quedan artefactos)
    filtered_src = re.sub(r"\\\s+\n", "\n", filtered_src)
    # 3) Normaliza finales de línea a \n
    filtered_src = filtered_src.replace("\r\n", "\n")

    # --- SANITIZADOR PARA EVITAR SyntaxError POR "\" DE CONTINUACIÓN ---
    # 1) Quita continuaciones de línea con backslash al final
    filtered_src = re.sub(r"\\\r?\n", "\n", filtered_src)
    # 2) Quita backslashes sueltos seguidos de espacios (a veces quedan artefactos)
    filtered_src = re.sub(r"\\\s+\n", "\n", filtered_src)
    # 3) Normaliza finales de línea a \n
    filtered_src = filtered_src.replace("\r\n", "\n")


    # Create an isolated module-like namespace
    ns = {}
    # Provide a minimal safe builtins (still default to Python builtins)
    ns["__builtins__"] = builtins.__dict__
    # Some projects rely on torch/numpy availability. Import lazily inside try blocks.
    prelude = """
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    class _Dummy: pass
    nn = _Dummy()
    F = _Dummy()
try:
    import numpy as np
except Exception:
    np = None
"""
    exec(prelude, ns, ns)

    if filtered_src.strip():
        try:
            exec(filtered_src, ns, ns)
        except Exception as e:
            # If execution fails, still return whatever got defined
            ns["_introspection_error"] = repr(e)
    else:
        ns["_introspection_error"] = "No definitions/imports extracted from notebook."
    return ns

def pytest_addoption(parser):
    parser.addoption("--strict-nb", action="store_true", default=False,
                     help="Fail tests instead of skipping when notebook pieces are missing.")

import pytest

@pytest.fixture(scope="session")
def nb_namespace(request):
    ns = load_notebook_namespace()
    strict = request.config.getoption("--strict-nb")
    return ns, strict
