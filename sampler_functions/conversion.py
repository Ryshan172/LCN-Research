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
    Constraints are built into the CPTs when possible
    (so forward_sample already respects them).
    """
    
    model = BayesianNetwork(lcn["edges"])
    cpds = []

    for node, table in lcn["credal_sets"].items():
        parents = [src for src, dst in lcn["edges"] if dst == node]
        n_states = 2
        evidence_card = [2] * len(parents)

        values = []
        for parent_assignment, probs in table.items():
            # Default from interval
            p_true = pick_prob(probs["True"], policy)
            p_false = 1 - p_true

            # Enforce constraints at CPT level
            for constr in lcn.get("logical_constraints", []):
                if_node, if_val = list(constr["if"].items())[0]
                then_node, then_val = list(constr["then"].items())[0]

                if node == then_node and f"{if_node}={if_val}" in parent_assignment:
                    if then_val:
                        p_true, p_false = 1.0, 0.0
                    else:
                        p_true, p_false = 0.0, 1.0

            values.append([p_true, p_false])

        # transpose rows=states, cols=configs
        values = list(zip(*values))

        cpd = TabularCPD(
            variable=node,
            variable_card=n_states,
            values=values,
            evidence=parents if parents else None,
            evidence_card=evidence_card if parents else None
        )
        cpds.append(cpd)

    model.add_cpds(*cpds)
    return model


def enforce_constraints(samples: pd.DataFrame, lcn: dict):
    """
    Double-check after sampling to catch anything that slipped through 
    (e.g., if multiple constraints interact in ways you didn't encode in the CPT).
    Post-process samples to enforce logical constraints.
    """

    for constr in lcn.get("logical_constraints", []):
        if_node, if_val = list(constr["if"].items())[0]
        then_node, then_val = list(constr["then"].items())[0]

        mask = (samples[if_node] == if_val)
        samples.loc[mask, then_node] = then_val
    return samples
