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
import multiprocessing as mp
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

# A small, conservative whitelist of built-ins the agent will legitimately
# want in pandas expressions (e.g. str(x), round(x, 2), len(df)). Earlier
# versions gave the eval namespace NO built-ins at all ({"__builtins__": {}}),
# which is maximally safe but also blocks completely harmless calls — in
# practice this caused live queries like `str(r)` inside a list comprehension
# to fail with NameError, wasting a tool-call round-trip on nothing.
#
# Deliberately EXCLUDED even though they're "just functions": type, dir,
# vars, isinstance, issubclass, super, classmethod, staticmethod, property,
# hasattr — anything that enables introspection or class-hierarchy walking
# (the classic sandbox-escape vector, e.g. reaching a class's subclasses).
# None of these are needed for ordinary pandas analysis expressions.
SAFE_BUILTINS = {
    "str": str, "int": int, "float": float, "bool": bool,
    "len": len, "round": round, "min": min, "max": max, "sum": sum,
    "sorted": sorted, "list": list, "dict": dict, "tuple": tuple, "set": set,
    "abs": abs, "range": range, "enumerate": enumerate, "zip": zip,
    "all": all, "any": any,
}

MAX_RESULT_CHARS = 1000

# The AST checks above stop code-injection (imports, dunder access,
# format-string escapes), but a syntactically "safe" expression can still
# be computationally hostile — e.g. building a huge array or looping
# forever inside a comprehension. Neither of those trips any AST rule, so
# they're bounded here instead: the query runs in an isolated child
# process with a wall-clock timeout and a memory cap, and that process is
# killed outright if it exceeds either. This only protects the sandboxed
# query itself, not the main app process.
QUERY_TIMEOUT_SECONDS = 5
QUERY_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024  # 512 MB

# multiprocessing's "fork" start method is what makes this cheap: on
# Linux/macOS, os.fork() gives the child a copy-on-write view of the
# parent's memory (including `df`), so the dataframe doesn't need to be
# pickled/copied for every single query the agent runs. "fork" isn't
# available on Windows — see the fallback in safe_query().
#
# Trade-off worth knowing: forking from a multi-threaded process (which
# Streamlit's server is) carries a small, well-documented risk of the
# child deadlocking if fork() lands while another thread holds a C-level
# lock (e.g. inside malloc). We accept this here because the forked
# child does one tiny, fast thing (eval one expression, send a string
# back, exit) and is hard-killed by the timeout regardless — a stuck
# child costs at most QUERY_TIMEOUT_SECONDS, it doesn't hang the app. The
# alternative, "forkserver", avoids that risk but loses the copy-on-write
# sharing of `df`, meaning it would re-pickle the whole dataframe on
# every single query — a bad trade for a tool the agent calls up to 5
# times per report. If this sandbox were handling much larger dataframes
# or higher query volume, forkserver (accepting the pickling cost) would
# be the safer default; revisit if that changes.
_FORK_AVAILABLE = "fork" in mp.get_all_start_methods()


def _run_isolated(safe_globals: dict, safe_locals: dict, tree: ast.AST, conn) -> None:
    """Runs in the forked child process. Applies a best-effort memory cap
    (POSIX only, via `resource`), evaluates the expression, and sends the
    formatted result string back over the pipe. Any crash of this process
    (OOM-killed, hit the memory cap, etc.) is handled by the parent, which
    is watching the process rather than blocking on this function."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        cap = QUERY_MEMORY_LIMIT_BYTES
        if hard != resource.RLIM_INFINITY:
            cap = min(cap, hard)
        resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
    except Exception:
        pass  # e.g. resource module unavailable on this platform

    try:
        result = eval(compile(tree, "<agent_query>", mode="eval"), safe_globals, safe_locals)
        conn.send(("ok", _format_result(result)))
    except MemoryError:
        conn.send(("error", "Query exceeded the memory limit and was terminated."))
    except Exception as e:
        conn.send(("error", f"Query failed with error: {type(e).__name__}: {e}"))
    finally:
        conn.close()


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
    safe_globals = {"__builtins__": SAFE_BUILTINS}
    safe_locals = {"df": df, "pd": pd, "np": np}

    if not _FORK_AVAILABLE:
        # No fork on this platform (e.g. Windows): fall back to running
        # inline. Still fully protected against code-injection by the AST
        # checks above, just without the timeout/memory ceiling.
        try:
            result = eval(compile(tree, "<agent_query>", mode="eval"), safe_globals, safe_locals)
        except Exception as e:
            return f"Query failed with error: {type(e).__name__}: {e}"
        result_str = _format_result(result)
        if len(result_str) > MAX_RESULT_CHARS:
            result_str = result_str[:MAX_RESULT_CHARS] + f"\n... [truncated, {len(result_str)} total chars]"
        return result_str

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_run_isolated, args=(safe_globals, safe_locals, tree, child_conn))
    proc.start()
    child_conn.close()  # only the child writes to this end

    proc.join(QUERY_TIMEOUT_SECONDS)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return f"Query exceeded the {QUERY_TIMEOUT_SECONDS}s time limit and was terminated."

    if parent_conn.poll():
        status, payload = parent_conn.recv()
    else:
        # Process died without sending anything back — almost always the
        # OS OOM-killer or the memory cap being hit hard enough to crash
        # the process rather than raise a catchable MemoryError.
        return "Query terminated unexpectedly, likely for exceeding the memory limit."

    if status == "error":
        return payload

    result_str = payload
    if len(result_str) > MAX_RESULT_CHARS:
        result_str = result_str[:MAX_RESULT_CHARS] + f"\n... [truncated, {len(result_str)} total chars]"
    return result_str


def _format_result(result) -> str:
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.to_string(max_rows=12)
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

    # Resource-exhaustion demo: syntactically safe, no AST rule objects to
    # either of these, but both are computationally hostile. Timeout is
    # lowered here just so this smoke test finishes quickly.
    print("\n--- Resource-exhaustion guard (safe syntax, hostile computation) ---")
    globals()["QUERY_TIMEOUT_SECONDS"] = 0.2  # lowered so this demo finishes quickly
    print(safe_query(df, "sum(range(10**9))"))
    globals()["QUERY_MEMORY_LIMIT_BYTES"] = 20 * 1024 * 1024  # 20 MB
    print(safe_query(df, "np.zeros(10**8)"))  # ~800 MB, well over the cap
