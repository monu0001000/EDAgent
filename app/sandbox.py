"""
sandbox.py
A restricted pandas-expression evaluator used by the agentic insight
generator (agent.py). This is the safety-critical piece: we're letting an
LLM decide what code to run against real data, so we do NOT use exec() or
give it real Python. Instead:

1. Only single expressions are allowed (no statements, no assignment,
   no imports, no function/class definitions) — enforced by using eval()
   with a restricted AST check, not exec().
2. The only names available in the eval namespace are `df`, `pd`, and `np`
   — no `__builtins__`, no `os`, `sys`, `subprocess`, `open`, etc.
3. We statically walk the parsed AST before evaluating anything, and
   reject the query if it contains dunder attribute access, imports,
   or any call to a small blocklist of dangerous names.
4. Result size is capped (both for token budget reasons and to prevent
   the tool being used to exfiltrate the whole raw dataset row-by-row).
"""

import ast
import pandas as pd
import numpy as np


class UnsafeQueryError(Exception):
    pass


BLOCKED_NAMES = {
    "exec", "eval", "compile", "open", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "input", "breakpoint", "format",
}

# Method names that can be used to reach attributes indirectly (format-string
# style attribute access, e.g. "{0.__class__}".format(x)) without any literal
# dunder syntax appearing in the AST as an Attribute node.
BLOCKED_METHOD_NAMES = {"format", "format_map"}

MAX_RESULT_CHARS = 3000


def _check_ast_safety(tree: ast.AST) -> None:
    """Walk the parsed expression and reject anything resembling an escape attempt."""
    for node in ast.walk(tree):
        # No imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeQueryError("Imports are not allowed.")
        # No dunder attribute access (blocks __class__, __globals__, etc.)
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeQueryError(f"Access to '{node.attr}' is not allowed.")
        # No assignment / statements — eval() with mode="eval" already
        # enforces this at parse time, but double-check defensively.
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            raise UnsafeQueryError("Assignment statements are not allowed.")
        # Blocked function names
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise UnsafeQueryError(f"Use of '{node.id}' is not allowed.")
        # Block calls to dunder-named functions via Call(func=Attribute(...))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr.startswith("__"):
                raise UnsafeQueryError("Dunder method calls are not allowed.")
            # Block .format()/.format_map() entirely — these let a string
            # literal's format spec reach dunder attributes (e.g.
            # "{0.__class__}".format(x)) without any Attribute node in the
            # AST containing literal dunder syntax, bypassing the check above.
            if node.func.attr in BLOCKED_METHOD_NAMES:
                raise UnsafeQueryError(f"'{node.func.attr}' is not allowed (can be used to bypass attribute checks).")
        # Defense in depth: reject any string literal that itself contains
        # dunder-looking text, in case some other indirect mechanism is found.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value:
                raise UnsafeQueryError("String literals containing '__' are not allowed.")
        # f-strings can also embed format specs that reach attributes
        if isinstance(node, ast.JoinedStr):
            raise UnsafeQueryError("f-strings are not allowed.")


def safe_query(df: pd.DataFrame, code: str) -> str:
    """
    Safely evaluate a single pandas expression against `df`.
    Returns a string representation of the result (truncated if large).
    Raises UnsafeQueryError if the code fails the safety check.
    """
    code = code.strip()
    if not code:
        raise UnsafeQueryError("Empty query.")

    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as e:
        raise UnsafeQueryError(f"Only single expressions are allowed (no statements). Parse error: {e}")

    _check_ast_safety(tree)

    # Deliberately minimal namespace: no __builtins__, so things like
    # open(), __import__(), etc. are unreachable even if the AST check
    # somehow missed a novel bypass.
    safe_globals = {"__builtins__": {}}
    safe_locals = {"df": df, "pd": pd, "np": np}

    try:
        result = eval(compile(tree, "<agent_query>", mode="eval"), safe_globals, safe_locals)
    except Exception as e:
        return f"Query failed with error: {type(e).__name__}: {e}"

    result_str = _format_result(result)
    if len(result_str) > MAX_RESULT_CHARS:
        result_str = result_str[:MAX_RESULT_CHARS] + f"\n... [truncated, {len(result_str)} total chars]"
    return result_str


def _format_result(result) -> str:
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.to_string(max_rows=30)
    return str(result)


if __name__ == "__main__":
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "x", "y", "z"]})

    # Should succeed
    print("--- Safe queries ---")
    print(safe_query(df, "df['a'].mean()"))
    print(safe_query(df, "df.groupby('b')['a'].sum()"))
    print(safe_query(df, "df[df['a'] > 2]"))

    # Should be blocked
    print("\n--- Unsafe queries (should all raise/be rejected) ---")
    unsafe_examples = [
        "__import__('os').system('echo hacked')",
        "open('/etc/passwd').read()",
        "df.__class__.__bases__",
        "exec('import os')",
        "df.a = 999",  # assignment
        "[x for x in ().__class__.__base__.__subclasses__()]",
    ]
    for q in unsafe_examples:
        try:
            r = safe_query(df, q)
            print(f"NOT BLOCKED (bug!): {q!r} -> {r}")
        except UnsafeQueryError as e:
            print(f"Blocked correctly: {q!r} -> {e}")
