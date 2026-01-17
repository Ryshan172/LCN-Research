import numpy as np

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
