"""
Serialization of Python expression graphs to a flat format for the Julia C ABI.

The compiled Julia library (MOI + GeneratorOptInterface + Bridges + HiGHS)
receives models as flat arrays through ctypes. This module converts the Python
expression tree into that flat representation.

Tags:
    0 = constant        -> [0, float64_value]
    1 = variable        -> [1, var_index]
    2 = binary op       -> [2, op_code, left..., right...]
    3 = unary op        -> [3, op_code, arg...]
    4 = function        -> [4, func_code, arg...]
    5 = iterator        -> [5, iterator_id]
    6 = indexed_var     -> [6, var_block_start, var_block_count, index_expr...]
    7 = indexed_param   -> [7, param_id, index_expr...]

Op codes for binary: + = 0, - = 1, * = 2, / = 3, ^ = 4
Op codes for unary:  - = 0
Func codes: sin = 0, cos = 1, exp = 2, log = 3, sqrt = 4, abs = 5
"""

from __future__ import annotations

from jumpy.expressions import (
    BinaryOp,
    Constant,
    Constraint,
    Expr,
    Func,
    IndexedParameter,
    IndexedVariable,
    UnaryOp,
    Variable,
)
from jumpy.iterators import Iterator

# Tag constants
TAG_CONST = 0
TAG_VAR = 1
TAG_BINARY = 2
TAG_UNARY = 3
TAG_FUNC = 4
TAG_ITERATOR = 5
TAG_INDEXED_VAR = 6
TAG_INDEXED_PARAM = 7

BINARY_OPS = {"+": 0, "-": 1, "*": 2, "/": 3, "^": 4}
UNARY_OPS = {"-": 0}
FUNC_CODES = {"sin": 0, "cos": 1, "exp": 2, "log": 3, "sqrt": 4, "abs": 5}


def serialize_expr(expr: Expr, param_registry: dict | None = None) -> list[float]:
    """
    Serialize an expression tree to a flat list of floats.

    This is the format passed across the ctypes boundary to the compiled
    Julia library, which reconstructs it into MOI.ScalarNonlinearFunction.
    """
    if param_registry is None:
        param_registry = {}
    buf: list[float] = []
    _serialize(expr, buf, param_registry)
    return buf


def _serialize(node: Expr, buf: list[float], param_registry: dict) -> None:
    match node:
        case Constant(value=v):
            buf.extend([TAG_CONST, v])
        case Variable(index=idx):
            buf.extend([TAG_VAR, float(idx)])
        case BinaryOp(op=op, left=left, right=right):
            buf.extend([TAG_BINARY, float(BINARY_OPS[op])])
            _serialize(left, buf, param_registry)
            _serialize(right, buf, param_registry)
        case UnaryOp(op=op, arg=arg):
            buf.extend([TAG_UNARY, float(UNARY_OPS[op])])
            _serialize(arg, buf, param_registry)
        case Func(name=name, arg=arg):
            buf.extend([TAG_FUNC, float(FUNC_CODES[name])])
            _serialize(arg, buf, param_registry)
        case Iterator() as it:
            buf.extend([TAG_ITERATOR, float(it.id)])
        case IndexedVariable() as iv:
            start = iv.variable_vector._variables[0].index
            count = len(iv.variable_vector)
            buf.extend([TAG_INDEXED_VAR, float(start), float(count)])
            _serialize(iv.index_expr, buf, param_registry)
        case IndexedParameter() as ip:
            param_id = id(ip.parameter)
            if param_id not in param_registry:
                param_registry[param_id] = {
                    "id": len(param_registry),
                    "values": ip.parameter.values,
                }
            buf.extend([TAG_INDEXED_PARAM, float(param_registry[param_id]["id"])])
            _serialize(ip.index_expr, buf, param_registry)
        case _:
            raise TypeError(f"Cannot serialize {type(node).__name__}")


def serialize_constraint(con: Constraint, param_registry: dict | None = None) -> dict:
    """
    Serialize a constraint: normalized expression + sense.

    Returns:
        {"expr": [...flat...], "sense": "<="|">="|"=="}
    """
    if param_registry is None:
        param_registry = {}
    normalized = con.lhs - con.rhs  # f(x) {<=,>=,==} 0
    return {
        "expr": serialize_expr(normalized, param_registry),
        "sense": con.sense,
    }
