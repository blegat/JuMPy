"""
Bridge between JuMPy's Python expression graph and MathOptInterface via juliacall.

Translates Python Expr nodes into Julia MOI function calls.
This module is only imported when backend="juliacall" is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jumpy.model import Model

from jumpy.expressions import (
    BinaryOp,
    Constant,
    Constraint,
    Func,
    IndexedParameter,
    IndexedVariable,
    UnaryOp,
    Variable,
)
from jumpy.iterators import Iterator


def build_moi_model(jl, model: Model) -> list[float]:
    """
    Build an MOI model in Julia from a JuMPy Model and solve it.

    For constraint groups, this constructs the GenOpt IteratedFunction
    representation so that expansion happens in Julia.

    For now (without GenOpt in juliacall), we fall back to expanding
    constraint groups in Python and adding them individually. This is
    slower but functionally correct — it lets users validate models
    before the juliac backend is ready.
    """
    jl.seval("""
        function _jumpy_create_optimizer()
            optimizer = MOI.instantiate(
                MOI.OptimizerWithAttributes(HiGHS.Optimizer, "output_flag" => false),
                with_bridge_type = Float64,
            )
            return optimizer
        end
    """)
    optimizer = jl._jumpy_create_optimizer()

    # Add variables
    jl_vars = []
    for block in model._var_blocks:
        for var in block.vector:
            jl_var = jl.MOI.add_variable(optimizer)
            jl_vars.append(jl_var)
            if block.lower is not None:
                jl.MOI.add_constraint(
                    optimizer, jl_var,
                    jl.MOI.GreaterThan(block.lower),
                )
            if block.upper is not None:
                jl.MOI.add_constraint(
                    optimizer, jl_var,
                    jl.MOI.LessThan(block.upper),
                )

    # Add constraint groups (expanded in Python for juliacall fallback)
    for group in model._constraint_groups:
        _add_constraint_group_expanded(jl, optimizer, jl_vars, group)

    # Add individual constraints
    for con in model._individual_constraints:
        _add_constraint(jl, optimizer, jl_vars, con)

    # Set objective
    if model._objective is not None:
        sense = (
            jl.MOI.MIN_SENSE
            if model._objective.sense == "min"
            else jl.MOI.MAX_SENSE
        )
        jl.MOI.set(optimizer, jl.MOI.ObjectiveSense(), sense)
        obj_func = _expr_to_moi(jl, jl_vars, model._objective.expr, {})
        jl.MOI.set(optimizer, jl.MOI.ObjectiveFunction(jl.typeof(obj_func)), obj_func)

    # Optimize
    jl.MOI.optimize_b(optimizer)

    # Extract solution
    solution = []
    for jl_var in jl_vars:
        val = float(jl.MOI.get(optimizer, jl.MOI.VariablePrimal(), jl_var))
        solution.append(val)

    return solution


def _add_constraint_group_expanded(jl, optimizer, jl_vars, group):
    """Expand a constraint group in Python and add each constraint to MOI."""
    from itertools import product

    ranges = [range(it.length) for it in group.iterators]
    for indices in product(*ranges):
        env = {}
        for it, idx in zip(group.iterators, indices):
            env[it.id] = it.values[idx]
        con = group.template
        _add_constraint_with_env(jl, optimizer, jl_vars, con, env)


def _add_constraint_with_env(jl, optimizer, jl_vars, con, env):
    """Add a single constraint, resolving iterator references from env."""
    lhs_func = _expr_to_moi(jl, jl_vars, con.lhs, env)
    rhs_func = _expr_to_moi(jl, jl_vars, con.rhs, env)

    # Normalize: lhs - rhs in set
    if con.sense == "<=":
        set_ = jl.MOI.LessThan(0.0)
    elif con.sense == ">=":
        set_ = jl.MOI.GreaterThan(0.0)
    elif con.sense == "==":
        set_ = jl.MOI.EqualTo(0.0)
    else:
        raise ValueError(f"Unknown constraint sense: {con.sense}")

    func = jl.MOI.ScalarNonlinearFunction(jl.Symbol("-"), jl.Any[lhs_func, rhs_func])
    jl.MOI.add_constraint(optimizer, func, set_)


def _expr_to_moi(jl, jl_vars, expr, env):
    """Convert a Python Expr to a Julia MOI function, resolving iterators from env."""
    match expr:
        case Constant(value=v):
            return v
        case Variable(index=idx):
            return jl_vars[idx]
        case BinaryOp(op=op, left=left, right=right):
            l = _expr_to_moi(jl, jl_vars, left, env)
            r = _expr_to_moi(jl, jl_vars, right, env)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol(op), jl.Any[l, r])
        case UnaryOp(op="-", arg=arg):
            a = _expr_to_moi(jl, jl_vars, arg, env)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol("-"), jl.Any[a])
        case Func(name=name, arg=arg):
            a = _expr_to_moi(jl, jl_vars, arg, env)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol(name), jl.Any[a])
        case Iterator() as it:
            return float(env[it.id])
        case IndexedVariable() as iv:
            idx_val = _eval_index(iv.index_expr, env)
            return jl_vars[iv.variable_vector._variables[0].index + int(idx_val)]
        case IndexedParameter() as ip:
            idx_val = _eval_index(ip.index_expr, env)
            return ip.parameter.values[int(idx_val)]
        case _:
            raise TypeError(f"Cannot convert {type(expr).__name__} to MOI")


def _eval_index(expr, env) -> float:
    """Evaluate an index expression with concrete iterator values from env."""
    match expr:
        case Constant(value=v):
            return v
        case Iterator() as it:
            return float(env[it.id])
        case BinaryOp(op=op, left=left, right=right):
            l = _eval_index(left, env)
            r = _eval_index(right, env)
            match op:
                case "+": return l + r
                case "-": return l - r
                case "*": return l * r
                case "/": return l / r
                case "^": return l ** r
        case _:
            raise TypeError(f"Cannot evaluate index expression: {type(expr).__name__}")
