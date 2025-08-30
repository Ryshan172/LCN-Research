# Collapse the LCN to a Bayesian Network and then sample 

import random
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import pandas as pd

def pick_prob(interval, policy="mid"):
    l, u = interval
    if policy == "lower": return l
    if policy == "upper": return u
    if policy == "mid":   return 0.5*(l+u)
    if policy == "random":return random.uniform(l, u)


def build_bn_from_lcn(lcn, policy="mid"):
    """
    Convert an LCN into a BayesianNetwork with pgmpy.
    Probabilities are taken from credal intervals using the chosen policy.
    Constraints are pushed into CPTs where possible.
    """
    model = BayesianNetwork(lcn["edges"])
    model.add_nodes_from(lcn["nodes"])  # Fix: include all nodes

    cpds = []

    for node, table in lcn["credal_sets"].items():
        parents = [src for src, dst in lcn["edges"] if dst == node]
        n_states = 2
        evidence_card = [2] * len(parents) if parents else None

        values = []
        for parent_assignment, probs in table.items():
            # Keep parent assignment standardized (strings)
            parent_assignment = parent_assignment.replace("1", "True").replace("0", "False")

            # Pick probability for True
            p_true = pick_prob(probs["True"], policy)
            p_false = 1 - p_true

            # Enforce CPT-level constraints
            for constr in lcn.get("logical_constraints", []):
                if_node, if_val = list(constr["if"].items())[0]
                then_node, then_val = list(constr["then"].items())[0]

                if_val = "True" if if_val else "False"
                then_val = "True" if then_val else "False"

                if node == then_node and f"{if_node}={if_val}" in parent_assignment:
                    if then_val == "True":
                        p_true, p_false = 1.0, 0.0
                    else:
                        p_true, p_false = 0.0, 1.0

            values.append([p_false, p_true])  # row order: False, True

        values = list(zip(*values))  # transpose for pgmpy

        cpd = TabularCPD(
            variable=node,
            variable_card=n_states,
            values=values,
            evidence=parents if parents else None,
            evidence_card=evidence_card if parents else None,
            state_names={node: ["False", "True"], **{p: ["False", "True"] for p in parents}}
        )
        cpds.append(cpd)

    model.add_cpds(*cpds)
    return model


def enforce_constraints(samples: pd.DataFrame, lcn: dict):
    """
    Post-process samples to enforce logical constraints.
    Ensures consistency when multiple constraints interact.
    """

    for constr in lcn.get("logical_constraints", []):
        if_node, if_val = list(constr["if"].items())[0]
        then_node, then_val = list(constr["then"].items())[0]

        # Force into "True"/"False"
        if_val = "True" if str(if_val) in ["1", "True"] else "False"
        then_val = "True" if str(then_val) in ["1", "True"] else "False"

        # Apply rule: if "if_node == if_val", enforce "then_node == then_val"
        mask = (samples[if_node] == if_val)
        samples.loc[mask, then_node] = then_val
    return samples
