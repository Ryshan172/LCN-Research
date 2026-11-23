from itertools import product
from math import fsum, prod

import numpy as np
from scipy.special import kl_div

def log_prob_sample(model, sample_dict):
    """
    Compute log P(x) for one sample (dict: var -> value)
    under a pgmpy BayesianModel.
    """
    logp = 0.0

    for cpd in model.get_cpds():
        node = cpd.variable
        parents = cpd.variables[1:]

        # Node value
        node_val = sample_dict[node]

        # Parent configuration
        parent_vals = {p: sample_dict[p] for p in parents}

        # Probability lookup
        prob = cpd.get_value(**{node: node_val, **parent_vals})

        logp += np.log(prob)

    return logp


def kl_divergence_from_samples(true_model, approx_model, samples_df):
    """
    Compute KL(P || Q) using existing forward samples from the true model.
    
    Parameters:
        true_model: pgmpy BayesianModel (P)
        approx_model: pgmpy BayesianModel (Q)
        samples_df: pandas DataFrame of samples from P

    Returns:
        float: KL divergence
    """

    print("Debugging")
    print("true_model", true_model)
    print("approx_model", approx_model)
    print("samples_df", samples_df)

    kl_sum = 0.0
    n = len(samples_df)

    for _, row in samples_df.iterrows():
        sample_dict = row.to_dict()
        logp = log_prob_sample(true_model, sample_dict)
        logq = log_prob_sample(approx_model, sample_dict)
        kl_sum += (logp - logq)

    return kl_sum / n


# Change done: enumerate the fitted joint distributions for the small RQ1 networks.
def _joint_masses(model, nodes):
    model.check_model()
    states = model.states
    cpds = model.get_cpds()
    masses = {}
    for values in product(*(states[node] for node in nodes)):
        assignment = dict(zip(nodes, values))
        masses[values] = prod(
            cpd.get_value(**{var: assignment[var] for var in cpd.variables})
            for cpd in cpds
        )
    return masses


def exact_kl_divergence(true_model, approx_model):
    """Compute KL(true_model || approx_model) in nats by complete enumeration."""
    nodes = tuple(true_model.nodes())
    if set(nodes) != set(approx_model.nodes()):
        raise ValueError("Exact KL requires both networks to contain the same variables.")
    p = _joint_masses(true_model, nodes)
    q = _joint_masses(approx_model, nodes)

    # kl_div computes p log(p/q) - p + q. The correction sums to zero for
    # normalized distributions. An omitted fitted state has probability zero.
    total = fsum(kl_div(p.get(world, 0.0), q.get(world, 0.0)) for world in p.keys() | q.keys())
    if total < -1e-14:
        raise ArithmeticError(f"Unexpected negative exact KL divergence: {total}")
    return 0.0 if total < 0.0 else total
