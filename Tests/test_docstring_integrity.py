"""Guards for editing docstrings by hand.

Translating comments across ~3000 lines removed the Chinese but twice broke the Python:
a one-line replacement landed in the middle of a multi-line docstring and left the
closing quotes above the text, and another moved continuation lines to column 0. Both
shut the whole module down -- ``import amr_sched.stream`` raised SyntaxError -- and the
first was only noticed because a test file failed to collect.

These tests do not check style. They check the two specific ways that edit went wrong,
because a file that does not import is not a file with a cosmetic problem.
"""
from __future__ import annotations

import ast
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _python_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       # "Material" is upstream RAWSim-O's Python 2 tooling, kept for
                       # reference and not part of this project's Python. It cannot be
                       # parsed by Python 3, so including it would only ever fail.
                       # "out" holds unpacked third-party tooling (a 3.2 GB Blender
                       # install, a downloaded CLI bundle). Its own addon scripts are not
                       # this project's Python, so scanning them would fail on someone
                       # else's code and say nothing about our edits.
                       if d not in {"results", "archive", "__pycache__", "obj", "bin",
                                    "node_modules", ".git", ".snapshots", "docs", "out",
                                    "Material", "RAWSimO.Core", "RAWSimO.MultiAgentRouting",
                                    "RAWSimO.SimulationStudio", "RAWSimO.Visualization"}]
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


@pytest.mark.parametrize("path", sorted(_python_files()), ids=lambda p: os.path.relpath(p, ROOT))
def test_every_module_parses(path):
    """A docstring edit must never leave a file unparseable."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    try:
        ast.parse(src)
    except SyntaxError as exc:
        pytest.fail("%s does not parse: %s (line %s)"
                    % (os.path.relpath(path, ROOT), exc.msg, exc.lineno))


def test_no_docstring_closes_before_its_body():
    """Catches the exact mistake: closing quotes above the explanatory text."""
    offenders = []
    for path in _python_files():
        with open(path, encoding="utf-8") as f:
            source = f.read()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                continue
            body = getattr(node, "body", None)
            if not body or not isinstance(body[0], ast.Expr):
                continue
            doc = body[0].value
            if not isinstance(doc, ast.Constant) or not isinstance(doc.value, str):
                continue
            # A docstring that ends with its first line while a bare expression or a
            # stray string follows is the signature of the broken edit.
            if len(body) > 1 and isinstance(body[1], ast.Expr) \
                    and isinstance(body[1].value, ast.Constant) \
                    and isinstance(body[1].value.value, str):
                offenders.append("%s:%d" % (os.path.relpath(path, ROOT), body[1].lineno))
    assert not offenders, (
        "a string literal follows the docstring, which is what a mis-spliced docstring "
        "looks like: " + ", ".join(offenders[:10]))
