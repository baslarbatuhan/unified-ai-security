"""tools/calculator.py — safe arithmetic evaluator.

**Never** uses Python's built-in `eval()` or `exec()`. We rely on
`simpleeval`, which parses the expression to an AST and walks it,
permitting only:

    * Numeric literals (int / float)
    * Arithmetic operators (+, -, *, /, %, **, parentheses)
    * Unary minus / plus

Anything else — `__import__`, attribute access, function calls, name
references — raises a parse-time error. This is **second-line**
defence: the gateway's allow-list regex
(`^[0-9+\\-*/().\\s]+$`) is the primary screen and rejects letters
outright. simpleeval is what saves us if the regex is ever loosened.

Example:
    >>> from tools import invoke
    >>> invoke("calc_evaluate", {"expression": "(2 + 3) * 4"})
    {'expression': '(2 + 3) * 4', 'result': 20}
    >>> invoke("calc_evaluate", {"expression": "__import__('os')"})
    {'error': "FeatureNotAvailable: ..."}  # simpleeval refuses

Optional dependency:
  * `simpleeval` (added to requirements). If absent, the module
    falls back to a tiny hand-rolled AST walker that supports the
    same subset — CI installs without the dep still work, demo runs
    use the maintained library.
"""
from __future__ import annotations

from typing import Any, Dict

from tools import _register


# Try the maintained library first; fall back to our minimal AST
# walker so a missing dep doesn't take the dashboard down. The
# fallback is *not* feature-equivalent — it's a safety net so CI
# can run without `pip install simpleeval`.
try:
    from simpleeval import simple_eval as _simple_eval

    def _evaluate(expr: str) -> Any:
        # Default whitelist is already restrictive (no functions, no
        # attribute access). We pass `names={}` to be explicit.
        return _simple_eval(expr, names={})

    _BACKEND = "simpleeval"

except ImportError:
    import ast
    import operator

    _BINOPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def _evaluate(expr: str) -> Any:  # type: ignore[misc]
        """Minimal AST walker. Permits only numeric literals + the
        operators above. Anything else → ValueError."""
        tree = ast.parse(expr, mode="eval")

        def _w(node):
            if isinstance(node, ast.Expression):
                return _w(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp):
                op = _BINOPS.get(type(node.op))
                if op is None:
                    raise ValueError(f"operator {type(node.op).__name__} not allowed")
                return op(_w(node.left), _w(node.right))
            if isinstance(node, ast.UnaryOp):
                op = _UNARY.get(type(node.op))
                if op is None:
                    raise ValueError(f"unary {type(node.op).__name__} not allowed")
                return op(_w(node.operand))
            raise ValueError(f"node {type(node).__name__} not allowed")

        return _w(tree)

    _BACKEND = "ast_fallback"


def call(*, expression: str, **_: Any) -> Dict[str, Any]:
    """Evaluate `expression` safely. Returns the numeric result or an
    error dict. Never raises."""
    expr = str(expression).strip()
    if not expr:
        return {"error": "empty expression"}
    try:
        value = _evaluate(expr)
    except Exception as exc:  # noqa: BLE001 — surface any parse/eval failure
        return {
            "expression": expr,
            "error": f"{type(exc).__name__}: {exc}",
            "backend": _BACKEND,
        }
    # Coerce numpy/Decimal etc. back to a JSON-friendly number.
    if isinstance(value, (int, float)):
        out_value: Any = value
    else:
        out_value = float(value)
    return {"expression": expr, "result": out_value, "backend": _BACKEND}


_register("calc_evaluate", call)


__all__ = ["call"]
