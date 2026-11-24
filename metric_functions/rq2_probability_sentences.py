"""Evaluate RQ2's binary LCN sentences under the fitted Bayesian network."""

import numpy as np
from pgmpy.factors import factor_product

from sampler_functions.bn_topological import parse_parent_config_string


def evaluate_probability_sentences(model, reference_lcn):
    """Check the fitted joint distribution against hard implications and credal rows.

    Result rates use only positive-support antecedents or conditioning events.
    Each supplied implication is one rule. Each reference table row is one
    interval check covering both Boolean outcomes, including unconditional rows.
    Maximum interval violation is measured in conditional probability units.
    """
    # Changed: Evaluate the fitted distribution after search, including worlds
    # absent from the training data. The largest RQ2 grid has 13 binary nodes.
    model.check_model()
    joint = factor_product(*(cpd.to_factor() for cpd in model.get_cpds()))
    mass = joint.values
    axes = {node: axis for axis, node in enumerate(joint.variables)}
    tolerance = 1e-12

    def event(assignment):
        """Index the joint table by a conjunction of Boolean literals."""
        index = [slice(None)] * mass.ndim
        for node, value in assignment.items():
            states = joint.state_names[node]
            # MLE can observe just one of the two possible Boolean states.
            index[axes[node]] = states.index(value) if value in states else slice(0, 0)
        return tuple(index)

    # Changed: Count a world that violates several implications only once.
    any_violation = np.zeros(mass.shape, dtype=bool)
    rules = []
    for rule in reference_lcn["logical_constraints"]:
        antecedent = rule["if"]
        consequent = rule["then"]
        violates = np.zeros(mass.shape, dtype=bool)
        violates[event(antecedent)] = True
        violates[event(consequent)] = False
        any_violation |= violates
        support = float(mass[event(antecedent)].sum())
        violation = float(mass[violates].sum())
        rules.append({
            "if": antecedent, "then": consequent,
            "antecedent_probability": support, "violation_probability": violation,
        })

    # Changed: Query the reference conditioning event in the fitted joint even
    # when the learned graph gives the child a different set of parents.
    rows = []
    for node, table in reference_lcn["credal_sets"].items():
        for label, bounds in table.items():
            condition = parse_parent_config_string(label)
            support = float(mass[event(condition)].sum())
            gap = None
            if support > 0:
                gap = 0.0
                for value in (True, False):
                    outcome = {**condition, node: value}
                    probability = float(mass[event(outcome)].sum()) / support
                    lower, upper = bounds[str(value)]
                    gap = max(gap, lower - probability, probability - upper)
            rows.append({"node": node, "condition": condition,
                         "condition_probability": support, "max_interval_violation": gap})

    supported_rules = [r for r in rules if r["antecedent_probability"] > 0]
    satisfied = sum(r["violation_probability"] <= tolerance for r in supported_rules)
    gaps = [r["max_interval_violation"] for r in rows if r["max_interval_violation"] is not None]
    within_bounds = sum(gap <= tolerance for gap in gaps)
    return {
        "summary": {
            "any_rule_violation_probability": float(mass[any_violation].sum()),
            "hard_rule_count": len(rules),
            "hard_rule_zero_support": len(rules) - len(supported_rules),
            "hard_rule_satisfied_supported": satisfied,
            "hard_rule_satisfaction_rate": satisfied / len(supported_rules) if supported_rules else None,
            "interval_row_count": len(rows),
            "interval_row_zero_support": len(rows) - len(gaps),
            "interval_row_satisfied_supported": within_bounds,
            "interval_row_satisfaction_rate": within_bounds / len(gaps) if gaps else None,
            "max_interval_violation": max(gaps, default=None),
        },
        "absolute_tolerance": tolerance,
        "hard_rules": rules,
        "interval_rows": rows,
    }
