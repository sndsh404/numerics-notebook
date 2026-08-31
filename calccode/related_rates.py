"""Related rates on symbolic relation trees.

The caller supplies a relation F(vars) = 0 as a symbolic.py tree, for
example x^2 + y^2 - 169 for a 13 m ladder. ``time_derivative``
differentiates both sides with respect to time by the chain rule:
dF/dt is the sum over variables of partial(F, var) * rate(var), where
each rate is a fresh Var named "<var>_dt". ``solve_rate`` substitutes
the known variable values and known rates with eval_multi and solves
the resulting equation for the one unknown rate.

The equation is always linear in the unknown rate, because the rate
appears once per term with no powers. So instead of solving
symbolically, solve_rate evaluates dF/dt twice: once with the unknown
rate set to 0 (giving the constant term b) and once with it set to 1
(giving a + b). The answer is -b / a.

Limits, stated plainly: one unknown rate per call, the relation must
hold at the instant in question, and the framework cannot set up the
relation for you. Choosing F and its variables is the actual hard
part of a related rates problem, and it stays with the caller.
"""

from __future__ import annotations

from calccode.symbolic import (
    Add,
    Const,
    Cos,
    Exp,
    Expr,
    Log,
    Mul,
    Pow,
    Sin,
    Var,
    eval_multi,
    partial,
)


def _var_names(expr: Expr) -> list[str]:
    """Variable names in a tree, in first-seen order, no repeats."""
    names: list[str] = []

    def walk(node: Expr) -> None:
        if isinstance(node, Const):
            return
        if isinstance(node, Var):
            if node.name not in names:
                names.append(node.name)
            return
        if isinstance(node, (Add, Mul)):
            walk(node.left)
            walk(node.right)
            return
        if isinstance(node, Pow):
            walk(node.base)
            return
        if isinstance(node, (Sin, Cos, Exp, Log)):
            walk(node.arg)
            return
        raise TypeError(f"cannot read variables from {type(node).__name__}")

    walk(expr)
    return names


def time_derivative(relation: Expr) -> Expr:
    """dF/dt of a relation tree: sum over vars of F_var * var_dt.

    Each rate is a Var named "<var>_dt", so the result is a tree that
    eval_multi can evaluate once values and rates are known.
    """
    names = _var_names(relation)
    if not names:
        raise ValueError("relation has no variables to differentiate")
    terms = [
        Mul(partial(relation, name), Var(f"{name}_dt"))
        for name in names
    ]
    total = terms[0]
    for term in terms[1:]:
        total = Add(total, term)
    return total.simplify()


def solve_rate(
    relation: Expr,
    values: dict[str, float],
    rates: dict[str, float],
    unknown: str,
) -> float:
    """Solve dF/dt = 0 for one unknown rate.

    ``values`` maps every variable name in the relation to its value
    at the instant in question. ``rates`` maps every variable except
    ``unknown`` to its known rate of change. The unknown rate comes
    back as a float. Raises ValueError if the unknown rate drops out
    of the equation entirely (its partial is zero at this instant).
    """
    if unknown in rates:
        raise ValueError(f"{unknown!r} is given a rate but also named unknown")
    names = _var_names(relation)
    if unknown not in names:
        raise ValueError(f"{unknown!r} does not appear in the relation")
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"no value given for variable(s) {missing}")
    missing_rates = [name for name in names if name != unknown and name not in rates]
    if missing_rates:
        raise ValueError(f"no rate given for variable(s) {missing_rates}")

    dFdt = time_derivative(relation)

    def env(unknown_rate: float) -> dict[str, float]:
        env = dict(values)
        for name in names:
            env[f"{name}_dt"] = unknown_rate if name == unknown else rates[name]
        return env

    constant_term = eval_multi(dFdt, env(0.0))
    coefficient = eval_multi(dFdt, env(1.0)) - constant_term
    if coefficient == 0.0:
        raise ValueError(
            f"the rate of {unknown!r} drops out of dF/dt at this instant; "
            "the partial with respect to it is zero"
        )
    return -constant_term / coefficient
