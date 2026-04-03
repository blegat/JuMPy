"""
Bridge between JuMPy's Python expression graph and MathOptInterface via juliacall.

Translates Python Expr nodes into Julia MOI + GenOpt types.
This module is only imported when backend="juliacall" is used.

The Python side does NO iteration over constraints or variables.
It builds one expression template per constraint group, hands it to GenOpt
as a FunctionGenerator, and lets Julia handle all expansion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jumpy.model import Model

from jumpy.expressions import (
    BinaryOp,
    Constant,
    Func,
    IndexedParameter,
    IndexedVariable,
    UnaryOp,
    Variable,
)
from jumpy.iterators import Iterator


_HELPERS_DEFINED = False


def _define_helpers(jl):
    """Define Julia helper functions once."""
    global _HELPERS_DEFINED
    if _HELPERS_DEFINED:
        return
    jl.seval("""
        function _jumpy_add_variables!(optimizer, count, lower, upper)
            vars = MOI.add_variables(optimizer, count)
            if !isnothing(lower)
                for v in vars
                    MOI.add_constraint(optimizer, v, MOI.GreaterThan(lower))
                end
            end
            if !isnothing(upper)
                for v in vars
                    MOI.add_constraint(optimizer, v, MOI.LessThan(upper))
                end
            end
            return vars
        end

        function _jumpy_add_constraint_group!(optimizer, func, sense)
            n = prod(length.(func.iterators))
            if sense == "<="
                set = MOI.Nonpositives(n)
            elseif sense == ">="
                set = MOI.Nonnegatives(n)
            elseif sense == "=="
                set = MOI.Zeros(n)
            else
                error("Unknown sense: $sense")
            end
            MOI.add_constraint(optimizer, func, set)
        end

        function _jumpy_make_generator(func, iterators)
            return GenOpt.FunctionGenerator{typeof(func)}(func, iterators)
        end

        function _jumpy_create_optimizer()
            return MOI.instantiate(
                MOI.OptimizerWithAttributes(HiGHS.Optimizer, "output_flag" => false),
                with_bridge_type = Float64,
            )
        end

        function _jumpy_get_solution(optimizer, vars)
            return [MOI.get(optimizer, MOI.VariablePrimal(), v) for v in vars]
        end
    """)
    _HELPERS_DEFINED = True


def build_moi_model(jl, model: Model) -> list[float]:
    """
    Build an MOI model in Julia from a JuMPy Model and solve it.

    Constraint groups are passed as GenOpt.FunctionGenerator objects
    so that expansion happens entirely in Julia.
    """
    _define_helpers(jl)

    optimizer = jl._jumpy_create_optimizer()

    # Add variables — one bulk call per block
    all_jl_vars = []
    for block in model._var_blocks:
        lower = float(block.lower) if block.lower is not None else jl.nothing
        upper = float(block.upper) if block.upper is not None else jl.nothing
        block_vars = jl._jumpy_add_variables_b(
            optimizer, block.count, lower, upper,
        )
        all_jl_vars.append(block_vars)

    # Add constraint groups via GenOpt
    for group in model._constraint_groups:
        _add_constraint_group(jl, optimizer, all_jl_vars, model, group)

    # Add individual constraints
    for con in model._individual_constraints:
        _add_individual_constraint(jl, optimizer, all_jl_vars, con)

    # Set objective
    if model._objective is not None:
        sense = (
            jl.MOI.MIN_SENSE
            if model._objective.sense == "min"
            else jl.MOI.MAX_SENSE
        )
        jl.MOI.set(optimizer, jl.MOI.ObjectiveSense(), sense)
        obj_func = _expr_to_moi(jl, all_jl_vars, model._objective.expr)
        obj_type = jl.typeof(obj_func)
        jl.MOI.set(optimizer, jl.MOI.ObjectiveFunction(obj_type), obj_func)

    # Optimize and extract solution
    jl.MOI.optimize_b(optimizer)

    # Flatten all variable blocks into one solution vector
    all_vars_flat = jl.seval("vcat")(*(v for v in all_jl_vars))
    jl_solution = jl._jumpy_get_solution(optimizer, all_vars_flat)
    return [float(jl_solution[i]) for i in range(1, len(jl_solution) + 1)]


def _add_constraint_group(jl, optimizer, all_jl_vars, model, group):
    """
    Add a constraint group as a single GenOpt.FunctionGenerator.

    Python builds the template expression and iterator list, then hands
    them to GenOpt. No Python-side iteration over constraint instances.
    """
    # Build GenOpt iterators
    genopt_iterators = jl.seval("GenOpt.Iterator[]")
    iter_id_map = {}
    for idx, it in enumerate(group.iterators):
        jl_values = jl.seval("collect")(it.values)
        jl_it = jl.GenOpt.Iterator(jl_values)
        jl.push_b(genopt_iterators, jl_it)
        iter_id_map[it.id] = idx + 1  # 1-based

    # Normalize: lhs - rhs in {Nonpositives, Nonnegatives, Zeros}
    normalized = group.template.lhs - group.template.rhs

    # Build MOI.ScalarNonlinearFunction template with GenOpt placeholders
    template_func = _expr_to_moi_template(
        jl, all_jl_vars, model, normalized, genopt_iterators, iter_id_map,
    )

    # Wrap in FunctionGenerator and add constraint — all in Julia
    func_gen = jl._jumpy_make_generator(template_func, genopt_iterators)
    jl._jumpy_add_constraint_group_b(optimizer, func_gen, group.template.sense)


def _get_jl_var(jl, all_jl_vars, var_index, model):
    """Get the Julia MOI.VariableIndex for a Python Variable by its index."""
    offset = 0
    for block_idx, block in enumerate(model._var_blocks):
        if var_index < offset + block.count:
            local_idx = var_index - offset
            return all_jl_vars[block_idx][local_idx + 1]  # 1-based
        offset += block.count
    raise IndexError(f"Variable index {var_index} out of range")


def _get_contiguous(jl, all_jl_vars, variable_vector, model):
    """Get a GenOpt.ContiguousArrayOfVariables for a VariableVector."""
    start = variable_vector._variables[0].index
    count = len(variable_vector)
    return jl.seval(
        f"GenOpt.ContiguousArrayOfVariables({start}, ({count},))"
    )


def _expr_to_moi_template(jl, all_jl_vars, model, expr, genopt_iterators, iter_id_map):
    """
    Convert a Python Expr into an MOI.ScalarNonlinearFunction template
    with GenOpt.IteratorIndex and ContiguousArrayOfVariables placeholders.
    """
    match expr:
        case Constant(value=v):
            return v
        case Variable(index=idx):
            return _get_jl_var(jl, all_jl_vars, idx, model)
        case BinaryOp(op=op, left=left, right=right):
            l = _expr_to_moi_template(jl, all_jl_vars, model, left, genopt_iterators, iter_id_map)
            r = _expr_to_moi_template(jl, all_jl_vars, model, right, genopt_iterators, iter_id_map)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol(op), jl.Any[l, r])
        case UnaryOp(op="-", arg=arg):
            a = _expr_to_moi_template(jl, all_jl_vars, model, arg, genopt_iterators, iter_id_map)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol("-"), jl.Any[a])
        case Func(name=name, arg=arg):
            a = _expr_to_moi_template(jl, all_jl_vars, model, arg, genopt_iterators, iter_id_map)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol(name), jl.Any[a])
        case Iterator() as it:
            jl_idx = iter_id_map[it.id]
            return jl.GenOpt.IteratorIndex(jl_idx)
        case IndexedVariable() as iv:
            contiguous = _get_contiguous(jl, all_jl_vars, iv.variable_vector, model)
            index_expr = _expr_to_moi_template(
                jl, all_jl_vars, model, iv.index_expr, genopt_iterators, iter_id_map,
            )
            # 0-based Python → 1-based Julia
            index_1based = jl.MOI.ScalarNonlinearFunction(
                jl.Symbol("+"), jl.Any[index_expr, 1],
            )
            return jl.MOI.ScalarNonlinearFunction(
                jl.Symbol("getindex"), jl.Any[contiguous, index_1based],
            )
        case IndexedParameter() as ip:
            jl_values = jl.seval("collect")(ip.parameter.values)
            index_expr = _expr_to_moi_template(
                jl, all_jl_vars, model, ip.index_expr, genopt_iterators, iter_id_map,
            )
            index_1based = jl.MOI.ScalarNonlinearFunction(
                jl.Symbol("+"), jl.Any[index_expr, 1],
            )
            return jl.MOI.ScalarNonlinearFunction(
                jl.Symbol("getindex"), jl.Any[jl_values, index_1based],
            )
        case _:
            raise TypeError(f"Cannot convert {type(expr).__name__} to MOI template")


def _expr_to_moi(jl, all_jl_vars, expr):
    """Convert a Python Expr to a concrete Julia MOI function (no iterators)."""
    match expr:
        case Constant(value=v):
            return v
        case Variable(index=idx):
            # For objectives/individual constraints, find the variable
            offset = 0
            for block_idx, block_vars in enumerate(all_jl_vars):
                block_size = int(jl.length(block_vars))
                if idx < offset + block_size:
                    return block_vars[idx - offset + 1]  # 1-based
                offset += block_size
            raise IndexError(f"Variable index {idx} out of range")
        case BinaryOp(op=op, left=left, right=right):
            l = _expr_to_moi(jl, all_jl_vars, left)
            r = _expr_to_moi(jl, all_jl_vars, right)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol(op), jl.Any[l, r])
        case UnaryOp(op="-", arg=arg):
            a = _expr_to_moi(jl, all_jl_vars, arg)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol("-"), jl.Any[a])
        case Func(name=name, arg=arg):
            a = _expr_to_moi(jl, all_jl_vars, arg)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol(name), jl.Any[a])
        case _:
            raise TypeError(f"Cannot convert {type(expr).__name__} to MOI")


def _add_individual_constraint(jl, optimizer, all_jl_vars, con):
    """Add a single non-grouped constraint."""
    lhs_func = _expr_to_moi(jl, all_jl_vars, con.lhs)
    rhs_func = _expr_to_moi(jl, all_jl_vars, con.rhs)

    if con.sense == "<=":
        set_ = jl.MOI.LessThan(0.0)
    elif con.sense == ">=":
        set_ = jl.MOI.GreaterThan(0.0)
    elif con.sense == "==":
        set_ = jl.MOI.EqualTo(0.0)
    else:
        raise ValueError(f"Unknown constraint sense: {con.sense}")

    func = jl.MOI.ScalarNonlinearFunction(
        jl.Symbol("-"), jl.Any[lhs_func, rhs_func],
    )
    jl.MOI.add_constraint(optimizer, func, set_)
