"""Tests for the JuMPy expression graph, symbolic indexing, and serialization."""

import sys
sys.path.insert(0, "src")

from jumpy import (
    Model, Iterator, Parameter, Variable, Constant,
    sin, exp, minimize, maximize, sum_over,
)
from jumpy.expressions import BinaryOp, Constraint, IndexedVariable, IndexedParameter
from jumpy.serialize import (
    serialize_expr,
    serialize_constraint,
    TAG_CONST,
    TAG_VAR,
    TAG_BINARY,
    TAG_FUNC,
    TAG_ITERATOR,
    TAG_INDEXED_VAR,
    TAG_INDEXED_PARAM,
)


# -- Expression building -------------------------------------------------------

def test_basic_arithmetic():
    x = Variable(0, "x")
    y = Variable(1, "y")
    expr = x + 2 * y
    assert isinstance(expr, BinaryOp)
    assert repr(expr) == "(x + (2.0 * y))"


def test_constraint():
    x = Variable(0, "x")
    y = Variable(1, "y")
    con = x + y <= 10
    assert isinstance(con, Constraint)
    assert con.sense == "<="


def test_nonlinear():
    x = Variable(0, "x")
    expr = sin(x) + exp(x)
    assert repr(expr) == "(sin(x) + exp(x))"


def test_negation():
    x = Variable(0, "x")
    expr = -x
    assert repr(expr) == "(-x)"


# -- Iterator as Expr (the core idea) ------------------------------------------

def test_iterator_in_arithmetic():
    """Iterator objects participate in arithmetic to build index expressions."""
    i = Iterator(range(10))
    j = Iterator(range(10))

    expr = 10 * i + j
    assert isinstance(expr, BinaryOp)
    assert expr.op == "+"
    assert isinstance(expr.left, BinaryOp)
    assert expr.left.op == "*"


def test_symbolic_variable_indexing():
    """x[i] with an Iterator produces an IndexedVariable."""
    m = Model()
    x = m.variables(100, lower=0, name="x")
    i = Iterator(range(99))

    indexed = x[i]
    assert isinstance(indexed, IndexedVariable)


def test_symbolic_variable_index_arithmetic():
    """x[i + 1] builds an index expression graph."""
    m = Model()
    x = m.variables(100, lower=0, name="x")
    i = Iterator(range(99))

    indexed = x[i + 1]
    assert isinstance(indexed, IndexedVariable)
    assert isinstance(indexed.index_expr, BinaryOp)


def test_multidim_index():
    """x[10*i + j] builds a compound index expression."""
    m = Model()
    x = m.variables(100, lower=0, name="x")
    i = Iterator(range(10))
    j = Iterator(range(10))

    indexed = x[10 * i + j]
    assert isinstance(indexed, IndexedVariable)


def test_parameter_symbolic_indexing():
    """costs[i] produces an IndexedParameter."""
    costs = Parameter([3.0, 1.5, 2.0], name="costs")
    i = Iterator(range(3))

    indexed = costs[i]
    assert isinstance(indexed, IndexedParameter)


def test_parameter_concrete_indexing():
    """costs[0] returns a Constant."""
    costs = Parameter([3.0, 1.5, 2.0])
    c = costs[0]
    assert isinstance(c, Constant)
    assert c.value == 3.0


# -- Full constraint group expressions -----------------------------------------

def test_constraint_group_expression():
    """The full expression x[i] + x[i+1] <= 10 builds a valid tree."""
    m = Model()
    x = m.variables(100, lower=0, name="x")
    i = Iterator(range(99))

    con = x[i] + x[i + 1] <= 10
    assert isinstance(con, Constraint)
    assert con.sense == "<="
    assert isinstance(con.lhs, BinaryOp)  # x[i] + x[i+1]
    assert isinstance(con.lhs.left, IndexedVariable)
    assert isinstance(con.lhs.right, IndexedVariable)


def test_nonlinear_constraint_group():
    """sin(x[i]) + exp(x[i]) <= 1.0 with symbolic indexing."""
    m = Model()
    x = m.variables(100, lower=0, name="x")
    i = Iterator(range(100))

    con = sin(x[i]) + exp(x[i]) <= 1.0
    assert isinstance(con, Constraint)


def test_parameter_in_constraint_group():
    """costs[i] * x[i] <= 50 mixes data and variables."""
    m = Model()
    x = m.variables(100, lower=0, name="x")
    costs = Parameter([float(k) for k in range(100)], name="costs")
    i = Iterator(range(100))

    con = costs[i] * x[i] <= 50
    assert isinstance(con, Constraint)
    assert isinstance(con.lhs, BinaryOp)
    assert con.lhs.op == "*"
    assert isinstance(con.lhs.left, IndexedParameter)
    assert isinstance(con.lhs.right, IndexedVariable)


# -- Model API -----------------------------------------------------------------

def test_model_constraint_group():
    m = Model()
    x = m.variables(100, lower=0, name="x")
    i = Iterator(range(99))

    group = m.constraint_group([i], x[i] + x[i + 1] <= 10)
    assert len(m._constraint_groups) == 1
    assert "99 constraints" in repr(group)


def test_model_multidim_constraint_group():
    m = Model()
    x = m.variables(100, lower=0, name="x")
    i = Iterator(range(10))
    j = Iterator(range(10))

    group = m.constraint_group([i, j], x[10 * i + j] >= 0)
    assert len(m._constraint_groups) == 1
    assert "100 constraints" in repr(group)


def test_model_objective():
    m = Model()
    x = m.variables(3, name="x")
    m.objective = minimize(x[0] + x[1] + x[2])
    assert m.objective.sense == "min"


# -- Serialization -------------------------------------------------------------

def test_serialize_constant():
    assert serialize_expr(Constant(3.14)) == [TAG_CONST, 3.14]


def test_serialize_variable():
    assert serialize_expr(Variable(42)) == [TAG_VAR, 42.0]


def test_serialize_binary():
    x = Variable(0)
    y = Variable(1)
    buf = serialize_expr(x + y)
    assert buf == [TAG_BINARY, 0.0, TAG_VAR, 0.0, TAG_VAR, 1.0]


def test_serialize_iterator():
    i = Iterator(range(10))
    buf = serialize_expr(i)
    assert buf == [TAG_ITERATOR, float(i.id)]


def test_serialize_indexed_variable():
    m = Model()
    x = m.variables(100, lower=0)
    i = Iterator(range(99))

    buf = serialize_expr(x[i])
    assert buf[0] == TAG_INDEXED_VAR
    assert buf[1] == 0.0   # start index
    assert buf[2] == 100.0  # count
    assert TAG_ITERATOR in buf


def test_serialize_indexed_parameter():
    costs = Parameter([1.0, 2.0, 3.0])
    i = Iterator(range(3))

    reg = {}
    buf = serialize_expr(costs[i], reg)
    assert buf[0] == TAG_INDEXED_PARAM
    assert len(reg) == 1


def test_serialize_full_model():
    """Serialize a complete model and verify the structure."""
    m = Model()
    x = m.variables(100, lower=0, name="x")

    i = Iterator(range(99))
    m.constraint_group([i], x[i] + x[i + 1] <= 10)

    costs = Parameter([float(k) for k in range(100)], name="costs")
    j = Iterator(range(100))
    m.constraint_group([j], costs[j] * x[j] <= 50)

    m.objective = minimize(x[0] + x[1])

    data = m._serialize()
    assert data["num_vars"] == 100
    assert len(data["var_blocks"]) == 1
    assert len(data["constraint_groups"]) == 2
    assert data["constraint_groups"][0]["iterators"][0]["length"] == 99
    assert data["constraint_groups"][1]["iterators"][0]["length"] == 100
    assert len(data["parameters"]) == 1  # costs
    assert data["objective"]["sense"] == "min"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL {test.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    import sys
    sys.exit(1 if failed else 0)
