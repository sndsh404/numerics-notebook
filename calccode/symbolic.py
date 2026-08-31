"""Symbolic differentiation over hand-built expression trees.

Each node is a small class with three methods: ``diff`` applies one
calculus rule and returns a new tree, ``eval`` computes a number, and
``simplify`` folds constants and strips identities like x + 0. Quotients
are written as a product with a negative power, so there is no separate
quotient rule.
"""

from __future__ import annotations

import math


class Expr:
    """Base class for every node in the expression tree."""

    def diff(self, var: str) -> "Expr":
        raise NotImplementedError

    def eval(self, x: float) -> float:
        raise NotImplementedError

    def simplify(self) -> "Expr":
        return self

    def __add__(self, other: "Expr | float") -> "Expr":
        return Add(self, _wrap(other))

    def __mul__(self, other: "Expr | float") -> "Expr":
        return Mul(self, _wrap(other))

    def __pow__(self, n: float) -> "Expr":
        return Pow(self, Const(float(n)))

    def __truediv__(self, other: "Expr | float") -> "Expr":
        return Mul(self, Pow(_wrap(other), Const(-1.0)))


def _wrap(value: "Expr | float") -> "Expr":
    return value if isinstance(value, Expr) else Const(float(value))


class Const(Expr):
    def __init__(self, value: float):
        self.value = float(value)

    def diff(self, var: str) -> Expr:
        return Const(0.0)

    def eval(self, x: float) -> float:
        return self.value

    def __str__(self) -> str:
        return str(int(self.value)) if self.value == int(self.value) else str(self.value)


class Var(Expr):
    def __init__(self, name: str = "x"):
        self.name = name

    def diff(self, var: str) -> Expr:
        return Const(1.0 if self.name == var else 0.0)

    def eval(self, x: float) -> float:
        return x

    def __str__(self) -> str:
        return self.name


class Add(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left, self.right = left, right

    def diff(self, var: str) -> Expr:
        return Add(self.left.diff(var), self.right.diff(var)).simplify()

    def eval(self, x: float) -> float:
        return self.left.eval(x) + self.right.eval(x)

    def simplify(self) -> Expr:
        left, right = self.left.simplify(), self.right.simplify()
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(left.value + right.value)
        if isinstance(left, Const) and left.value == 0.0:
            return right
        if isinstance(right, Const) and right.value == 0.0:
            return left
        return Add(left, right)

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


class Mul(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left, self.right = left, right

    def diff(self, var: str) -> Expr:
        # Product rule: (uv)' = u'v + uv'.
        return Add(
            Mul(self.left.diff(var), self.right),
            Mul(self.left, self.right.diff(var)),
        ).simplify()

    def eval(self, x: float) -> float:
        return self.left.eval(x) * self.right.eval(x)

    def simplify(self) -> Expr:
        left, right = self.left.simplify(), self.right.simplify()
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(left.value * right.value)
        for node, other in ((left, right), (right, left)):
            if isinstance(node, Const):
                if node.value == 0.0:
                    return Const(0.0)
                if node.value == 1.0:
                    return other
        return Mul(left, right)

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


class Pow(Expr):
    """A base raised to a constant exponent."""

    def __init__(self, base: Expr, exponent: Const):
        if not isinstance(exponent, Const):
            raise TypeError("Pow only supports constant exponents")
        self.base, self.exponent = base, exponent

    def diff(self, var: str) -> Expr:
        # Power rule with chain rule: (u^n)' = n u^(n-1) u'.
        n = self.exponent.value
        return Mul(
            Mul(Const(n), Pow(self.base, Const(n - 1.0))),
            self.base.diff(var),
        ).simplify()

    def eval(self, x: float) -> float:
        return self.base.eval(x) ** self.exponent.value

    def simplify(self) -> Expr:
        base = self.base.simplify()
        if self.exponent.value == 0.0:
            return Const(1.0)
        if self.exponent.value == 1.0:
            return base
        if isinstance(base, Const):
            return Const(base.value ** self.exponent.value)
        return Pow(base, self.exponent)

    def __str__(self) -> str:
        return f"({self.base}^{self.exponent})"


class _Unary(Expr):
    fn: staticmethod
    name: str

    def __init__(self, arg: Expr):
        self.arg = arg

    def eval(self, x: float) -> float:
        return float(self.fn(self.arg.eval(x)))

    def simplify(self) -> Expr:
        arg = self.arg.simplify()
        if isinstance(arg, Const):
            return Const(float(self.fn(arg.value)))
        return type(self)(arg)

    def __str__(self) -> str:
        return f"{self.name}({self.arg})"


class Sin(_Unary):
    fn = staticmethod(math.sin)
    name = "sin"

    def diff(self, var: str) -> Expr:
        return Mul(Cos(self.arg), self.arg.diff(var)).simplify()


class Cos(_Unary):
    fn = staticmethod(math.cos)
    name = "cos"

    def diff(self, var: str) -> Expr:
        return Mul(Mul(Const(-1.0), Sin(self.arg)), self.arg.diff(var)).simplify()


class Exp(_Unary):
    fn = staticmethod(math.exp)
    name = "exp"

    def diff(self, var: str) -> Expr:
        return Mul(Exp(self.arg), self.arg.diff(var)).simplify()


class Log(_Unary):
    fn = staticmethod(math.log)
    name = "log"

    def diff(self, var: str) -> Expr:
        # (log u)' = u' / u, written as u' times u^(-1).
        return Mul(self.arg.diff(var), Pow(self.arg, Const(-1.0))).simplify()


def diff(expr: Expr, var: str = "x") -> Expr:
    """Differentiate an expression tree and simplify the result."""
    return expr.diff(var).simplify()


def partial(expr: Expr, var: str) -> Expr:
    """Partial derivative with respect to one variable.

    Var.diff already returns 0 for any other name, so partials are the
    same machinery as diff, called with the variable of interest.
    """
    return expr.diff(var).simplify()


def implicit_diff(dFdx: Expr, dFdy: Expr) -> Expr:
    """dy/dx for a curve defined by F(x, y) = 0, given the partials.

    The caller computes the partials first, usually with partial(F, "x")
    and partial(F, "y") on a tree that holds both a Var("x") and a
    Var("y"). This returns the tree for -Fx / Fy, written as a product
    with Fy to the power -1 to match the rest of the module. The trees
    stay symbolic; eval() only handles one variable, so use eval_multi
    with values for both names to get a number out.
    """
    return Mul(Mul(Const(-1.0), dFdx), Pow(dFdy, Const(-1.0))).simplify()


def eval_multi(expr: Expr, env: dict[str, float]) -> float:
    """Evaluate a tree with one value per variable name.

    Single-variable eval() cannot do this because every Var reads the
    same x. This walks the tree and looks each Var up in env instead.
    """
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        if expr.name not in env:
            raise KeyError(f"no value given for variable {expr.name!r}")
        return env[expr.name]
    if isinstance(expr, Add):
        return eval_multi(expr.left, env) + eval_multi(expr.right, env)
    if isinstance(expr, Mul):
        return eval_multi(expr.left, env) * eval_multi(expr.right, env)
    if isinstance(expr, Pow):
        return eval_multi(expr.base, env) ** expr.exponent.value
    if isinstance(expr, _Unary):
        return float(expr.fn(eval_multi(expr.arg, env)))
    raise TypeError(f"cannot evaluate node of type {type(expr).__name__}")
