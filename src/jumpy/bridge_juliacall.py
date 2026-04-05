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
_any_vec = None


def _define_helpers(jl):
    """Define Julia helper functions once."""
    global _HELPERS_DEFINED, _any_vec
    if _HELPERS_DEFINED:
        return
    # jl.Any[...] is broken in PythonCall with Julia 1.12+
    _any_vec = jl.seval("(args...) -> Any[args...]")
    jl.seval("""
        function _jumpy_scalar_affine(terms, constant)
            return MOI.ScalarAffineFunction(terms, constant)
        end

        function _jumpy_affine_term(coef, var)
            return MOI.ScalarAffineTerm(coef, var)
        end

        function _jumpy_set_objective!(optimizer, sense, func)
            MOI.set(optimizer, MOI.ObjectiveSense(), sense)
            F = typeof(func)
            MOI.set(optimizer, MOI.ObjectiveFunction{F}(), func)
        end
    """)
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

        function _jumpy_make_generator(func, iterators, target_type)
            return GenOpt.FunctionGenerator{target_type}(func, iterators)
        end

        function _jumpy_create_optimizer()
            optimizer = MOI.instantiate(
                MOI.OptimizerWithAttributes(HiGHS.Optimizer, "output_flag" => false),
                with_bridge_type = Float64,
            )
            MOI.Bridges.add_bridge(optimizer, GenOpt.FunctionGeneratorBridge{Float64})
            return optimizer
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
        obj_func = _expr_to_moi(jl, all_jl_vars, model._objective.expr)
        jl._jumpy_set_objective_b(optimizer, sense, obj_func)

    # Optimize and extract solution
    jl.MOI.optimize_b(optimizer)

    # Flatten all variable blocks into one solution vector
    all_vars_flat = jl.seval("vcat")(*(v for v in all_jl_vars))
    jl_solution = jl._jumpy_get_solution(optimizer, all_vars_flat)
    return [float(jl_solution[i]) for i in range(len(jl_solution))]


def _is_linear_template(expr) -> bool:
    """Check if a template expression is linear (no nonlinear functions)."""
    match expr:
        case Constant() | Variable() | Iterator() | IndexedVariable() | IndexedParameter():
            return True
        case BinaryOp(op=op, left=left, right=right):
            if op in ("+", "-", "*"):
                return _is_linear_template(left) and _is_linear_template(right)
            return False
        case UnaryOp(op="-", arg=arg):
            return _is_linear_template(arg)
        case Func():
            return False
        case _:
            return False


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

    # Determine target function type: affine if template is linear, else nonlinear
    if _is_linear_template(group.template.lhs) and _is_linear_template(group.template.rhs):
        target_type = jl.seval("MOI.ScalarAffineFunction{Float64}")
    else:
        target_type = jl.seval("MOI.ScalarNonlinearFunction")

    # Wrap in FunctionGenerator and add constraint — all in Julia
    func_gen = jl._jumpy_make_generator(template_func, genopt_iterators, target_type)
    jl._jumpy_add_constraint_group_b(optimizer, func_gen, group.template.sense)


def _get_jl_var(jl, all_jl_vars, var_index, model):
    """Get the Julia MOI.VariableIndex for a Python Variable by its index."""
    offset = 0
    for block_idx, block in enumerate(model._var_blocks):
        if var_index < offset + block.count:
            local_idx = var_index - offset
            return all_jl_vars[block_idx][local_idx]  # PythonCall uses 0-based indexing
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
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol(op), _any_vec(l, r))
        case UnaryOp(op="-", arg=arg):
            a = _expr_to_moi_template(jl, all_jl_vars, model, arg, genopt_iterators, iter_id_map)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol("-"), _any_vec(a))
        case Func(name=name, arg=arg):
            a = _expr_to_moi_template(jl, all_jl_vars, model, arg, genopt_iterators, iter_id_map)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol(name), _any_vec(a))
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
                jl.Symbol("+"), _any_vec(index_expr, 1),
            )
            return jl.MOI.ScalarNonlinearFunction(
                jl.Symbol("getindex"), _any_vec(contiguous, index_1based),
            )
        case IndexedParameter() as ip:
            jl_values = jl.seval("collect")(ip.parameter.values)
            index_expr = _expr_to_moi_template(
                jl, all_jl_vars, model, ip.index_expr, genopt_iterators, iter_id_map,
            )
            index_1based = jl.MOI.ScalarNonlinearFunction(
                jl.Symbol("+"), _any_vec(index_expr, 1),
            )
            return jl.MOI.ScalarNonlinearFunction(
                jl.Symbol("getindex"), _any_vec(jl_values, index_1based),
            )
        case _:
            raise TypeError(f"Cannot convert {type(expr).__name__} to MOI template")


def _get_jl_var_by_index(jl, all_jl_vars, idx):
    """Get Julia MOI.VariableIndex for a Python variable by global index."""
    offset = 0
    for block_idx, block_vars in enumerate(all_jl_vars):
        block_size = int(jl.length(block_vars))
        if idx < offset + block_size:
            return block_vars[idx - offset]  # PythonCall uses 0-based indexing
        offset += block_size
    raise IndexError(f"Variable index {idx} out of range")


def _collect_linear_terms(expr, terms, sign=1.0):
    """
    Try to decompose expr into linear terms: list of (coef, var_index) + constant.
    Returns (success, constant).
    """
    match expr:
        case Constant(value=v):
            return True, v * sign
        case Variable(index=idx):
            terms.append((sign, idx))
            return True, 0.0
        case BinaryOp(op="+", left=left, right=right):
            terms_before = len(terms)
            ok_l, const_l = _collect_linear_terms(left, terms, sign)
            if not ok_l:
                del terms[terms_before:]
                return False, 0.0
            ok_r, const_r = _collect_linear_terms(right, terms, sign)
            if not ok_r:
                del terms[terms_before:]
                return False, 0.0
            return True, const_l + const_r
        case BinaryOp(op="-", left=left, right=right):
            terms_before = len(terms)
            ok_l, const_l = _collect_linear_terms(left, terms, sign)
            if not ok_l:
                del terms[terms_before:]
                return False, 0.0
            ok_r, const_r = _collect_linear_terms(right, terms, -sign)
            if not ok_r:
                del terms[terms_before:]
                return False, 0.0
            return True, const_l + const_r
        case BinaryOp(op="*", left=Constant(value=v), right=right):
            return _collect_linear_terms(right, terms, sign * v)
        case BinaryOp(op="*", left=left, right=Constant(value=v)):
            return _collect_linear_terms(left, terms, sign * v)
        case UnaryOp(op="-", arg=arg):
            return _collect_linear_terms(arg, terms, -sign)
        case _:
            return False, 0.0


def _expr_to_moi_linear(jl, all_jl_vars, expr):
    """
    Try to convert expr to ScalarAffineFunction. Returns None if nonlinear.
    """
    terms = []
    ok, constant = _collect_linear_terms(expr, terms)
    if not ok:
        return None

    jl_terms = jl.seval("MOI.ScalarAffineTerm{Float64}[]")
    for coef, var_idx in terms:
        jl_var = _get_jl_var_by_index(jl, all_jl_vars, var_idx)
        jl.push_b(jl_terms, jl._jumpy_affine_term(float(coef), jl_var))

    return jl._jumpy_scalar_affine(jl_terms, float(constant))


def _expr_to_moi(jl, all_jl_vars, expr):
    """Convert a Python Expr to a concrete Julia MOI function (no iterators)."""
    # Try linear first
    linear = _expr_to_moi_linear(jl, all_jl_vars, expr)
    if linear is not None:
        return linear

    match expr:
        case Constant(value=v):
            return v
        case Variable(index=idx):
            return _get_jl_var_by_index(jl, all_jl_vars, idx)
        case BinaryOp(op=op, left=left, right=right):
            l = _expr_to_moi(jl, all_jl_vars, left)
            r = _expr_to_moi(jl, all_jl_vars, right)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol(op), _any_vec(l, r))
        case UnaryOp(op="-", arg=arg):
            a = _expr_to_moi(jl, all_jl_vars, arg)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol("-"), _any_vec(a))
        case Func(name=name, arg=arg):
            a = _expr_to_moi(jl, all_jl_vars, arg)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol(name), _any_vec(a))
        case _:
            raise TypeError(f"Cannot convert {type(expr).__name__} to MOI")


def _add_individual_constraint(jl, optimizer, all_jl_vars, con):
    """Add a single non-grouped constraint."""
    # Normalize: lhs - rhs in {set}
    from jumpy.expressions import BinaryOp as _BinaryOp, Constant as _Constant
    normalized = con.lhs - con.rhs

    func = _expr_to_moi(jl, all_jl_vars, normalized)

    if con.sense == "<=":
        set_ = jl.MOI.LessThan(0.0)
    elif con.sense == ">=":
        set_ = jl.MOI.GreaterThan(0.0)
    elif con.sense == "==":
        set_ = jl.MOI.EqualTo(0.0)
    else:
        raise ValueError(f"Unknown constraint sense: {con.sense}")

    jl.MOI.Utilities.normalize_and_add_constraint(optimizer, func, set_)
