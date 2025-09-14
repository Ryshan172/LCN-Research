import math
import pandas as pd

"""
Basic Implementation of Interval BIC Score 
- uses a range of log-likelihoods
- note: logical constraints haven't been integrated into the 
"""

def compute_loglikelihood_bounds(row):
    """
    Compute log-likelihood bounds (interval) for each row
    in the aggregate table dataset
    """

    N_total = row["N_total"]

    # Probabilities with lower and upper bounds
    p_false_lower = max(1e-9, row["count_false_lower"] / N_total)
    p_false_upper = max(1e-9, row["count_false_upper"] / N_total)
    p_true_lower  = max(1e-9, row["count_true_lower"]  / N_total)
    p_true_upper  = max(1e-9, row["count_true_upper"]  / N_total)

    # Worst-case (min log-likelihood)
    ll_min = (
        row["count_false_lower"] * math.log(p_false_lower) +
        row["count_true_lower"]  * math.log(p_true_lower)
    )

    # Best-case (max log-likelihood)
    ll_max = (
        row["count_false_upper"] * math.log(p_false_upper) +
        row["count_true_upper"]  * math.log(p_true_upper)
    )

    return ll_min, ll_max


def compute_interval_bic_score(aggregate_table):
    # Computer interval log-likelihoods for each row
    ll_min_total, ll_max_total = 0.0, 0.0
    for _, row in aggregate_table.iterrows():
        ll_min, ll_max = compute_loglikelihood_bounds(row)
        ll_min_total += ll_min
        ll_max_total += ll_max

    # BIC Calculation
    # Total samples
    M = aggregate_table["N_total"].sum()
    # Very rough: 1 parameter per row (can refine)
    dim_G = len(aggregate_table)
    penalty = (math.log(M) / 2.0) * dim_G

    bic_min = ll_min_total - penalty
    bic_max = ll_max_total - penalty

    print("Log-likelihood interval: ", (ll_min_total, ll_max_total))
    print("BIC interval: ", (bic_min, bic_max))
