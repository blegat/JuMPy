"""
Expression graph for JuMPy.

Builds a tree of nodes that maps directly to MOI.ScalarNonlinearFunction:
    ScalarNonlinearFunction(head::Symbol, args::Vector{Any})

Each node is either:
    - Variable(index)              -> MOI.VariableIndex(index)
    - Constant(value)              -> Float64
    - BinaryOp(op, l, r)          -> MOI.ScalarNonlinearFunction(op, [l, r])
    - UnaryOp(op, arg)            -> MOI.ScalarNonlinearFunction(op, [arg])
    - Func(name, arg)             -> MOI.ScalarNonlinearFunction(name, [arg])
    - IteratorRef(iterator)        -> GeneratorOptInterface.IteratorIndex
    - IndexedVariable(vec, index)  -> variable lookup resolved during expansion
    - IndexedParameter(param, idx) -> data lookup resolved during expansion
"""

from __future__ import annotations

from typing import Union

Numeric = Union[int, float]


def _wrap(other: Expr | Numeric) -> Expr:
    """Wrap a numeric literal into a Constant node."""
    if isinstance(other, Expr):
        return other
    if isinstance(other, (int, float)):
        return Constant(float(other))
    raise TypeError(f"Cannot convert {type(other).__name__} to Expr")


class Expr:
    """Base class for all expression-graph nodes."""

    # -- arithmetic operators --------------------------------------------------

    def __add__(self, other: Expr | Numeric) -> BinaryOp:
        return BinaryOp("+", self, _wrap(other))

    def __radd__(self, other: Expr | Numeric) -> BinaryOp:
        return BinaryOp("+", _wrap(other), self)

    def __sub__(self, other: Expr | Numeric) -> BinaryOp:
        return BinaryOp("-", self, _wrap(other))

    def __rsub__(self, other: Expr | Numeric) -> BinaryOp:
        return BinaryOp("-", _wrap(other), self)

    def __mul__(self, other: Expr | Numeric) -> BinaryOp:
        return BinaryOp("*", self, _wrap(other))

    def __rmul__(self, other: Expr | Numeric) -> BinaryOp:
        return BinaryOp("*", _wrap(other), self)

    def __truediv__(self, other: Expr | Numeric) -> BinaryOp:
        return BinaryOp("/", self, _wrap(other))

    def __rtruediv__(self, other: Expr | Numeric) -> BinaryOp:
        return BinaryOp("/", _wrap(other), self)

    def __pow__(self, other: Expr | Numeric) -> BinaryOp:
        return BinaryOp("^", self, _wrap(other))

    def __rpow__(self, other: Expr | Numeric) -> BinaryOp:
        return BinaryOp("^", _wrap(other), self)

    def __neg__(self) -> UnaryOp:
        return UnaryOp("-", self)

    def __pos__(self) -> Expr:
        return self

    # -- comparison operators (return Constraint objects) ----------------------

    def __le__(self, other: Expr | Numeric) -> Constraint:
        return Constraint(self, "<=", _wrap(other))

    def __ge__(self, other: Expr | Numeric) -> Constraint:
        return Constraint(self, ">=", _wrap(other))

    def __eq__(self, other: Expr | Numeric) -> Constraint:
        return Constraint(self, "==", _wrap(other))


class Variable(Expr):
    """
    A decision variable. Maps to MOI.VariableIndex(index).

    Users don't create these directly; they are returned by Model.variables().
    """

    def __init__(self, index: int, name: str | None = None):
        self.index = index
        self.name = name

    def __repr__(self) -> str:
        if self.name:
            return self.name
        return f"x[{self.index}]"


class Constant(Expr):
    """A numeric constant in the expression tree."""

    def __init__(self, value: float):
        self.value = value

    def __repr__(self) -> str:
        return str(self.value)


class BinaryOp(Expr):
    """Binary operation node: +, -, *, /, ^."""

    def __init__(self, op: str, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


class UnaryOp(Expr):
    """Unary operation node (currently just negation)."""

    def __init__(self, op: str, arg: Expr):
        self.op = op
        self.arg = arg

    def __repr__(self) -> str:
        return f"({self.op}{self.arg})"


class Func(Expr):
    """Named function call: sin, cos, exp, log, sqrt, abs."""

    def __init__(self, name: str, arg: Expr):
        self.name = name
        self.arg = _wrap(arg)

    def __repr__(self) -> str:
        return f"{self.name}({self.arg})"


class IndexedVariable(Expr):
    """
    Symbolic variable lookup: x[expr] where expr contains IteratorRefs.

    During expansion on the Julia side, the index expression is evaluated
    for each iterator value to resolve to a concrete MOI.VariableIndex.
    """

    def __init__(self, variable_vector: VariableVector, index_expr: Expr):
        self.variable_vector = variable_vector
        self.index_expr = index_expr

    def __repr__(self) -> str:
        name = self.variable_vector.name or "x"
        return f"{name}[{self.index_expr}]"


class IndexedParameter(Expr):
    """
    Symbolic data lookup: param[expr] where expr contains IteratorRefs.

    During expansion, the index expression is evaluated for each iterator
    value to look up a concrete float from the parameter data.
    """

    def __init__(self, parameter: Parameter, index_expr: Expr):
        self.parameter = parameter
        self.index_expr = index_expr

    def __repr__(self) -> str:
        name = self.parameter.name or "p"
        return f"{name}[{self.index_expr}]"


class VariableVector:
    """
    A block of decision variables returned by Model.variables().

    Supports both concrete indexing (x[0] -> Variable) and symbolic
    indexing (x[i] -> IndexedVariable, where i is an Iterator/Expr).
    """

    def __init__(self, variables: list[Variable], name: str | None = None):
        self._variables = variables
        self.name = name

    def __getitem__(self, index: int | Expr) -> Variable | IndexedVariable:
        if isinstance(index, (int,)):
            return self._variables[index]
        if isinstance(index, Expr):
            return IndexedVariable(self, index)
        raise TypeError(f"Index must be int or Expr, got {type(index).__name__}")

    def __len__(self) -> int:
        return len(self._variables)

    def __iter__(self):
        return iter(self._variables)

    def __repr__(self) -> str:
        name = self.name or "x"
        return f"{name}[0:{len(self._variables)}]"


class Parameter:
    """
    A vector of constant data for use in constraint group templates.

    Supports symbolic indexing: costs[i] where i is an Iterator.

    Example:
        costs = jp.Parameter([3.0, 1.5, 2.0])
        m.constraint_group([i], costs[i] * x[i] <= 50)
    """

    def __init__(self, values: list[float], name: str | None = None):
        self.values = [float(v) for v in values]
        self.name = name

    def __getitem__(self, index: int | Expr) -> Constant | IndexedParameter:
        if isinstance(index, (int,)):
            return Constant(self.values[index])
        if isinstance(index, Expr):
            return IndexedParameter(self, index)
        raise TypeError(f"Index must be int or Expr, got {type(index).__name__}")

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        name = self.name or "param"
        return f"{name}[0:{len(self.values)}]"


class Constraint:
    """
    Represents lhs {<=, >=, ==} rhs.

    Normalized before passing to MOI as: (lhs - rhs) in {Nonpositives, Nonnegatives, Zeros}.
    """

    def __init__(self, lhs: Expr, sense: str, rhs: Expr):
        self.lhs = lhs
        self.sense = sense
        self.rhs = rhs

    def __repr__(self) -> str:
        return f"{self.lhs} {self.sense} {self.rhs}"


class Objective:
    """An optimization objective (minimize or maximize)."""

    def __init__(self, sense: str, expr: Expr):
        self.sense = sense
        self.expr = expr

    def __repr__(self) -> str:
        return f"{self.sense}({self.expr})"


# -- convenience functions that return Func nodes -----------------------------

def sin(x: Expr | Numeric) -> Func:
    return Func("sin", _wrap(x))

def cos(x: Expr | Numeric) -> Func:
    return Func("cos", _wrap(x))

def exp(x: Expr | Numeric) -> Func:
    return Func("exp", _wrap(x))

def log(x: Expr | Numeric) -> Func:
    return Func("log", _wrap(x))

def sqrt(x: Expr | Numeric) -> Func:
    return Func("sqrt", _wrap(x))

def abs(x: Expr | Numeric) -> Func:
    return Func("abs", _wrap(x))
