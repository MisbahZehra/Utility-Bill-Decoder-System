"""Safe arithmetic evaluator so numeric answers are computed by code, not by the LLM."""
import ast
import operator as op

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Mod: op.mod, ast.FloorDiv: op.floordiv, ast.Pow: op.pow}
_UN = {ast.USub: op.neg, ast.UAdd: op.pos}


def _avg(*a):
    xs = list(a[0]) if len(a) == 1 and isinstance(a[0], (list, tuple)) else list(a)
    return sum(xs) / len(xs)


def _flat(a):
    return list(a[0]) if len(a) == 1 and isinstance(a[0], (list, tuple)) else list(a)


FUNCS = {
    "avg": _avg, "mean": _avg,
    "sum": lambda *a: sum(_flat(a)),
    "min": lambda *a: min(_flat(a)),
    "max": lambda *a: max(_flat(a)),
    "round": round, "abs": abs, "len": lambda x: len(x),
    "count_gt": lambda xs, t: sum(1 for x in xs if x > t),
    "count_lt": lambda xs, t: sum(1 for x in xs if x < t),
    "count_gte": lambda xs, t: sum(1 for x in xs if x >= t),
}


def _ev(n):
    if isinstance(n, ast.Expression):
        return _ev(n.body)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
        a, b = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Pow) and (abs(b) > 10 or abs(a) > 1e6):
            raise ValueError("power too large")
        return _BIN[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
        return _UN[type(n.op)](_ev(n.operand))
    if isinstance(n, (ast.List, ast.Tuple)):
        return [_ev(e) for e in n.elts]
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS and not n.keywords:
        return FUNCS[n.func.id](*[_ev(a) for a in n.args])
    raise ValueError("unsupported expression")


def safe_eval(expr: str):
    """Evaluate an arithmetic expression; returns a rounded number or raises ValueError."""
    if not isinstance(expr, str) or len(expr) > 500:
        raise ValueError("bad expression")
    try:
        val = _ev(ast.parse(expr.strip(), mode="eval"))
    except (SyntaxError, TypeError, ZeroDivisionError, OverflowError) as e:
        raise ValueError(str(e)) from e
    if isinstance(val, list):
        return val
    val = round(float(val), 4)
    return int(val) if val == int(val) else val
