"""
Basic JuMPy usage example.

Shows the complete user-facing API. All expression graphs are built once
in Python; GeneratorOptInterface expands them in compiled Julia.
"""

import sys
sys.path.insert(0, "src")

import jumpy as jp

# ── Create a model ────────────────────────────────────────────────────────────

m = jp.Model()
x = m.variables(100, lower=0, name="x")

# ── Constraint group: consecutive variable pairs ──────────────────────────────
# Instead of 99 individual constraints built in Python (slow!),
# we define ONE template. GenOpt expands it in compiled Julia.

i = jp.Iterator(range(99))
m.constraint_group([i], x[i] + x[i + 1] <= 10)

# ── Nonlinear constraint group ────────────────────────────────────────────────

j = jp.Iterator(range(100))
m.constraint_group([j], jp.sin(x[j]) + jp.exp(x[j]) <= 1.0)

# ── Multi-dimensional constraint group ────────────────────────────────────────

p = jp.Iterator(range(10))
q = jp.Iterator(range(10))
m.constraint_group([p, q], x[10 * p + q] >= 0)

# ── Constraint group with data parameters ─────────────────────────────────────

costs = jp.Parameter([float(k) * 0.5 for k in range(100)], name="costs")
k = jp.Iterator(range(100))
m.constraint_group([k], costs[k] * x[k] <= 50)

# ── Objective ─────────────────────────────────────────────────────────────────

m.objective = jp.minimize(x[0] + x[1] + x[2])

# ── Inspect ───────────────────────────────────────────────────────────────────

print(f"Variables: {m._num_vars}")
print(f"Constraint groups: {len(m._constraint_groups)}")
for idx, group in enumerate(m._constraint_groups):
    print(f"  Group {idx}: {group}")
print(f"Objective: {m.objective}")

# Serialize to see the flat representation
data = m._serialize()
print(f"\nSerialized:")
print(f"  {data['num_vars']} variables")
print(f"  {len(data['constraint_groups'])} constraint group(s)")
print(f"  {len(data['parameters'])} parameter array(s)")
print(f"  objective sense: {data['objective']['sense']}")

# m.optimize()  # ← calls compiled Julia library (not yet available)
