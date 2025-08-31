# Worflow forward sampling an LCN that has been collapsed into a Bayesian Network
from pgmpy.sampling import BayesianModelSampling
from .conversion import build_bn_from_lcn, enforce_constraints

def convert_and_sample(lcn):

    # Build BN with constraints in CPTs
    bn_model = build_bn_from_lcn(lcn, policy="mid")

    # Forward sample
    sampler = BayesianModelSampling(bn_model)
    samples = sampler.forward_sample(size=1000)

    # Post-process (safety check)
    samples = enforce_constraints(samples, lcn)

    print(samples.head())

    return samples
