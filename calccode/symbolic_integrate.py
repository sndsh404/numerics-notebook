"""Rule-based symbolic antiderivatives over the symbolic.py trees.

``integrate`` walks an expression tree and applies a small set of
rules: constants, the power rule (including x^(-1) giving Log), sin,
cos, exp, sums, and constant multiples. A narrow integration by parts
set covers x times sin, cos, or exp of x (and constant multiples), and
ln(x) through the classic 1 * ln(x) trick. One substitution pattern is
in: when the argument of sin, cos, or exp, or the base of a power, is
a linear a*x + b, the antiderivative is F(a*x + b) / a by the linear
u-substitution. Anything outside that set raises NotImplementedError
naming the subtree it gave up on; the matcher only fires on exact
pattern hits.

``definite_integral`` applies FTC part 2: evaluate the antiderivative
at the bounds and subtract. When no rule matches the integrand it
falls back to Simpson's rule from integrals.py, because a number is
still useful even when a formula is out of reach.
"""

from __future__ import annotations

from calccode.integrals import simpson
from calccode.symbolic import Add, Const, Cos, Exp, Expr, Log, Mul, Pow, Sin, Var, diff


def _is_var(node: Expr, var: str) -> bool:
    return isinstance(node, Var) and node.name == var


def _linear_coefficients(node: Expr, var: str) -> "tuple[float, float] | None":
    """Coefficients (a, b) if the tree is a*x + b, else None.

    Walks Const, Var, Add, and Mul by a constant and folds them into a
    slope and intercept. Anything else (a nonlinear subtree, a Var with
    a different name) returns None, so the caller falls through to
    NotImplementedError instead of guessing.
    """
    node = node.simplify()
    if isinstance(node, Const):
        return (0.0, node.value)
    if _is_var(node, var):
        return (1.0, 0.0)
    if isinstance(node, Add):
        left = _linear_coefficients(node.left, var)
        right = _linear_coefficients(node.right, var)
        if left is not None and right is not None:
            return (left[0] + right[0], left[1] + right[1])
        return None
    if isinstance(node, Mul):
        for const, rest in ((node.left, node.right), (node.right, node.left)):
            if isinstance(const, Const):
                inner = _linear_coefficients(rest, var)
                if inner is not None:
                    return (const.value * inner[0], const.value * inner[1])
        return None
    return None


def _integrate_by_parts_mul(expr: Mul, var: str) -> "Expr | None":
    """Integration by parts for x times sin(x), cos(x), or exp(x).

    Matches Mul in either factor order with u = x and dv one of the
    three transcendentals applied to the plain variable, then applies
    int(u dv) = u v - int(v du). The remaining integral of v du is
    v itself because du = 1, so one recursive integrate() call closes
    it. Returns None when the pattern does not match.
    """
    for u, dv_expr in ((expr.left, expr.right), (expr.right, expr.left)):
        if not _is_var(u, var):
            continue
        if isinstance(dv_expr, (Sin, Cos, Exp)) and _is_var(dv_expr.arg, var):
            v = integrate(dv_expr, var)
            inner = Mul(v, diff(u, var)).simplify()
            return Add(
                Mul(u, v),
                Mul(Const(-1.0), integrate(inner, var)),
            ).simplify()
    return None


def integrate(expr: Expr, var: str = "x") -> Expr:
    """Antiderivative of a tree with respect to one variable.

    Covers constants, Var, Add, constant multiples, Pow with constant
    exponent (n = -1 maps to Log), and Sin, Cos, Exp. The argument of
    Sin, Cos, Exp and the base of Pow may be the plain variable or a
    linear a*x + b, handled by the linear u-substitution. Integration
    by parts covers x times Sin, Cos, or Exp of the variable, and Log
    of the variable by the 1 * ln(x) trick. A Var with a different
    name is treated as a constant, matching how diff treats it. Raises
    NotImplementedError for everything else.
    """
    expr = expr.simplify()

    if isinstance(expr, Const):
        return Mul(expr, Var(var))

    if isinstance(expr, Var):
        if expr.name == var:
            return Mul(Const(0.5), Pow(Var(var), Const(2.0)))
        return Mul(expr, Var(var))

    if isinstance(expr, Add):
        return Add(integrate(expr.left, var), integrate(expr.right, var))

    if isinstance(expr, Mul):
        for const, rest in ((expr.left, expr.right), (expr.right, expr.left)):
            if isinstance(const, Const):
                return Mul(const, integrate(rest, var))
        by_parts = _integrate_by_parts_mul(expr, var)
        if by_parts is not None:
            return by_parts
        raise NotImplementedError(
            f"no antiderivative rule for product {expr}; "
            "only constant multiples and x times sin, cos, or exp of x "
            "are supported"
        )

    if isinstance(expr, Pow):
        n = expr.exponent.value
        if _is_var(expr.base, var):
            if n == -1.0:
                return Log(Var(var))
            return Mul(
                Const(1.0 / (n + 1.0)),
                Pow(Var(var), Const(n + 1.0)),
            )
        linear = _linear_coefficients(expr.base, var)
        if linear is not None and linear[0] != 0.0:
            # Linear u-substitution: int((a x + b)^n) = (a x + b)^(n+1) / (a (n+1)).
            a = linear[0]
            if n == -1.0:
                return Mul(Const(1.0 / a), Log(expr.base))
            return Mul(
                Const(1.0 / (a * (n + 1.0))),
                Pow(expr.base, Const(n + 1.0)),
            )
        raise NotImplementedError(
            f"no antiderivative rule for {expr}; "
            "power rule needs the plain variable or a linear a*x + b as base"
        )

    if isinstance(expr, (Sin, Cos, Exp)):
        if _is_var(expr.arg, var):
            if isinstance(expr, Sin):
                return Mul(Const(-1.0), Cos(Var(var)))
            if isinstance(expr, Cos):
                return Sin(Var(var))
            return Exp(Var(var))
        linear = _linear_coefficients(expr.arg, var)
        if linear is not None and linear[0] != 0.0:
            # Linear u-substitution: int f(a x + b) = F(a x + b) / a.
            a = linear[0]
            if isinstance(expr, Sin):
                outer = Mul(Const(-1.0), Cos(expr.arg))
            elif isinstance(expr, Cos):
                outer = Sin(expr.arg)
            else:
                outer = Exp(expr.arg)
            return Mul(Const(1.0 / a), outer)
        raise NotImplementedError(
            f"no antiderivative rule for {expr}; "
            "the argument must be the plain variable or a linear a*x + b"
        )

    if isinstance(expr, Log):
        if not _is_var(expr.arg, var):
            raise NotImplementedError(
                f"no antiderivative rule for {expr}; "
                "the argument must be the plain variable"
            )
        # By parts with u = ln(x), dv = 1 dx: int(ln x) = x ln x - int(x * 1/x).
        # The inner integrand x * x^(-1) does not fold to 1 in this tree,
        # so the known result x is written out directly.
        return Add(
            Mul(Var(var), Log(Var(var))),
            Mul(Const(-1.0), Var(var)),
        ).simplify()

    raise NotImplementedError(
        f"no antiderivative rule for subtree {expr} "
        f"of type {type(expr).__name__}"
    )


def definite_integral(
    expr: Expr,
    a: float,
    b: float,
    var: str = "x",
    panels: int = 1000,
) -> float:
    """Definite integral from a to b.

    Tries the symbolic antiderivative first and subtracts endpoint
    values (FTC part 2). If no rule matches, falls back to Simpson's
    rule from integrals.py with ``panels`` panels and returns that
    approximation instead. The fallback needs eval() on the integrand,
    so it only works when every Var in the tree reads the same x.
    """
    try:
        antiderivative = integrate(expr, var)
    except NotImplementedError:
        return simpson(expr.eval, a, b, panels)
    return antiderivative.eval(b) - antiderivative.eval(a)
