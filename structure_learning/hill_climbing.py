from pgmpy.estimators import HillClimbSearch, BIC
import os
from pgmpy.estimators import StructureScore

# Prevent joblib from using multiprocessing (the cause of the signal error)
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["JOBLIB_START_METHOD"] = "threading"

class ConstantScore(StructureScore):
    """
    A minimal pgmpy StructureScore wrapper that always returns a fixed, precomputed score.

    This is used to plug a network-level interval BIC (or any scalar score) into
    HillClimbSearch.estimate() without computing per-model scores.

    - The candidate model passed to `score()` is ignored.
    - Useful when you already computed a global score for the entire network
      (e.g., lower bound, midpoint, or upper bound of interval BIC) and just
      want to run Hill Climbing using that scalar.
    """
    def __init__(self, data, score_value):
        super().__init__(data)
        self.score_value = float(score_value)

    def score(self, model):
        # Ignore the model, always return the precomputed score
        return self.score_value

    def local_score(self, node, parents):
        """
        pgmpy calls this method to compute the score contribution of each node
        given its parents. We ignore it here and return the same fixed score.
        """
        return self.score_value


def run_hillclimb_search(df):
    # Convert true/false columns to categorical values
    df = df.astype("category")
    
    # Feed to hillclimb
    hc = HillClimbSearch(df)
    best_model = hc.estimate(scoring_method=BIC(df))
    return (best_model.edges())


def run_hillclimbing_bic(df):
    df = df.astype("category")

    # Create the scoring method
    bic = BIC(df)

    # Run hill-climbing
    hc = HillClimbSearch(df)
    best_model = hc.estimate(scoring_method=bic)

    # Compute the BIC score of the final structure
    best_score = bic.score(best_model)

    return best_model.edges(), best_score


def run_interval_bic_hillclimb(df, score_value):
    """
    Run greedy hill-climbing using a fixed scalar score for the entire network.
    This version ignores per-node intervals because you already computed
    the network-level interval BIC.
    """

    df = df.astype("category")

    # Use the constant score wrapper
    score_fn = ConstantScore(df, score_value)

    hc = HillClimbSearch(df)
    best_model = hc.estimate(scoring_method=score_fn)
    best_score = score_fn.score(best_model)

    return best_model.edges(), best_score
