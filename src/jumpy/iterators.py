"""
Iterators for constraint groups.

An Iterator IS an Expr node. When used in arithmetic (10*i + j), it builds
an expression graph. When used as an index (x[i], costs[i]), it creates
symbolic IndexedVariable / IndexedParameter nodes.

Maps to GeneratorOptInterface.Iterator(length, values) on the Julia side.
"""

from __future__ import annotations

from jumpy.expressions import Expr


class Iterator(Expr):
    """
    An index set for constraint groups.

    When used in expressions, it acts as a symbolic placeholder that
    GeneratorOptInterface expands over its values during constraint generation.

    Example:
        i = jp.Iterator(range(99))
        j = jp.Iterator(range(10))

        x[i]            # symbolic variable lookup
        x[10*i + j]     # symbolic index arithmetic
        costs[i] * x[i] # symbolic data + variable lookup
    """

    _next_id: int = 0

    def __init__(self, values):
        self.values = list(values)
        self.length = len(self.values)
        self.id = Iterator._next_id
        Iterator._next_id += 1

    def __repr__(self) -> str:
        return f"i{self.id}"
