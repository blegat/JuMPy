"""
The Model class: top-level API for building optimization models in JuMPy.

A Model collects variables, constraint groups, and an objective, then hands
everything off to the compiled Julia library for expansion and solving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from jumpy.expressions import (
    Constraint,
    Expr,
    Objective,
    Parameter,
    Variable,
    VariableVector,
)
from jumpy.iterators import Iterator
from jumpy.serialize import serialize_constraint, serialize_expr
from jumpy.backend import Backend, get_backend


def minimize(expr: Expr) -> Objective:
    return Objective("min", expr)


def maximize(expr: Expr) -> Objective:
    return Objective("max", expr)


def sum_over(iterator: Iterator, expr: Expr) -> Expr:
    """
    Symbolic sum over an iterator.

    Passed to Julia as a SumNode that GenOpt expands, NOT evaluated in Python.

    Example:
        jp.sum_over(i, costs[i] * x[i])
    """
    return SumExpr(iterator, expr)


@dataclass
class ConstraintGroup:
    """
    A group of constraints defined by a template expression + iterators.

    Maps to GeneratorOptInterface.IteratedFunction:
        IteratedFunction(func::MOI.ScalarNonlinearFunction, iterators::Vector{Iterator})

    The template is a Constraint containing Iterator nodes as placeholders.
    Expansion happens entirely on the Julia side.
    """

    template: Constraint
    iterators: list[Iterator]

    def __repr__(self) -> str:
        n = 1
        for it in self.iterators:
            n *= it.length
        return f"ConstraintGroup({n} constraints from {len(self.iterators)} iterator(s))"


@dataclass
class VariableBlock:
    """A contiguous block of variables with shared bounds."""

    start: int
    count: int
    lower: float | None
    upper: float | None
    vector: VariableVector


class SumExpr(Expr):
    """Symbolic sum over an iterator. Expanded on the Julia side."""

    def __init__(self, iterator: Iterator, body: Expr):
        self.iterator = iterator
        self.body = body

    def __repr__(self) -> str:
        return f"sum({self.iterator}, {self.body})"


class Model:
    """
    A JuMPy optimization model.

    Example:
        m = jp.Model()
        x = m.variables(100, lower=0)

        i = jp.Iterator(range(99))
        m.constraint_group([i], x[i] + x[i + 1] <= 10)

        m.objective = jp.minimize(sum(x))
        m.optimize()
    """

    def __init__(self, backend: str = "juliac"):
        """
        Create a new model.

        Args:
            backend: "juliac" (default, no Julia needed) or "juliacall"
                     (uses juliacall, installs Julia lazily if needed).
        """
        self._backend: Backend = get_backend(backend)
        self._var_blocks: list[VariableBlock] = []
        self._constraint_groups: list[ConstraintGroup] = []
        self._individual_constraints: list[Constraint] = []
        self._objective: Objective | None = None
        self._num_vars: int = 0
        self._solution: list[float] | None = None

    # -- Variables -------------------------------------------------------------

    def variables(
        self,
        count: int,
        *,
        lower: float | None = None,
        upper: float | None = None,
        name: str | None = None,
    ) -> VariableVector:
        """
        Add a block of decision variables.

        Returns a VariableVector that supports both concrete (x[0])
        and symbolic (x[i]) indexing.
        """
        start = self._num_vars
        vars = []
        for k in range(count):
            var_name = f"{name}[{k}]" if name else None
            vars.append(Variable(start + k, var_name))
        self._num_vars += count
        vec = VariableVector(vars, name)
        self._var_blocks.append(VariableBlock(start, count, lower, upper, vec))
        return vec

    def variable(
        self,
        *,
        lower: float | None = None,
        upper: float | None = None,
        name: str | None = None,
    ) -> Variable:
        """Add a single decision variable."""
        vec = self.variables(1, lower=lower, upper=upper, name=name)
        return vec[0]

    # -- Constraints -----------------------------------------------------------

    def constraint_group(
        self,
        iterators: list[Iterator],
        template: Constraint,
    ) -> ConstraintGroup:
        """
        Add a constraint group.

        The template is an expression containing Iterator nodes as symbolic
        placeholders. GeneratorOptInterface expands the template over all
        iterator values entirely in compiled Julia.

        Example:
            i = jp.Iterator(range(99))
            m.constraint_group([i], x[i] + x[i + 1] <= 10)

            i = jp.Iterator(range(10))
            j = jp.Iterator(range(10))
            m.constraint_group([i, j], x[10*i + j] >= 0)
        """
        group = ConstraintGroup(template, iterators)
        self._constraint_groups.append(group)
        return group

    def constraint(self, con: Constraint) -> Constraint:
        """Add a single (non-grouped) constraint."""
        self._individual_constraints.append(con)
        return con

    # -- Objective -------------------------------------------------------------

    @property
    def objective(self) -> Objective | None:
        return self._objective

    @objective.setter
    def objective(self, obj: Objective) -> None:
        self._objective = obj

    # -- Solve -----------------------------------------------------------------

    def optimize(self) -> None:
        """
        Solve the model using the selected backend.

        - juliac backend: serializes to flat arrays, calls compiled shared library
        - juliacall backend: builds MOI model directly in Julia via juliacall
        """
        self._solution = self._backend.optimize(self)

    def _serialize(self) -> dict:
        """Serialize the entire model for the Julia C ABI."""
        param_registry: dict = {}
        data: dict = {
            "num_vars": self._num_vars,
            "var_blocks": [],
            "constraint_groups": [],
            "individual_constraints": [],
            "objective": None,
            "parameters": [],  # filled at the end from param_registry
        }

        for block in self._var_blocks:
            data["var_blocks"].append({
                "start": block.start,
                "count": block.count,
                "lower": block.lower,
                "upper": block.upper,
            })

        for group in self._constraint_groups:
            serialized_con = serialize_constraint(group.template, param_registry)
            iterators = []
            for it in group.iterators:
                iterators.append({
                    "id": it.id,
                    "length": it.length,
                    "values": [float(v) for v in it.values],
                })
            data["constraint_groups"].append({
                **serialized_con,
                "iterators": iterators,
            })

        for con in self._individual_constraints:
            data["individual_constraints"].append(
                serialize_constraint(con, param_registry)
            )

        if self._objective:
            data["objective"] = {
                "sense": self._objective.sense,
                "expr": serialize_expr(self._objective.expr, param_registry),
            }

        # Collect all referenced parameters
        params_by_id = sorted(param_registry.values(), key=lambda p: p["id"])
        data["parameters"] = [p["values"] for p in params_by_id]

        return data

    # -- Solution retrieval ----------------------------------------------------

    def value(self, var: Variable) -> float:
        """Get the solved value of a variable."""
        if self._solution is None:
            raise RuntimeError("Model has not been solved yet. Call optimize() first.")
        return self._solution[var.index]


