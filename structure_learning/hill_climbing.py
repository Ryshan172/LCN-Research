import networkx as nx
from pgmpy.base import DAG
from pgmpy.estimators import HillClimbSearch, BIC, ExpertKnowledge
import os
from pgmpy.estimators import StructureScore

from scoring_functions.interval_bic_derivation import compute_interval_BIC
from scoring_functions.scoring_helpers import regroup_for_parents

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


# Change done: compare legal first moves by actual scores when the initial IBIC is -inf.
def _rq1_start_dag(nodes, scorer, required_edges):
    dag = DAG()
    dag.add_nodes_from(nodes)
    dag.add_edges_from(required_edges)
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Required RQ1 rule edges form a cycle")
    unbounded = [
        node for node in nodes
        if scorer.local_score(node, list(dag.predecessors(node))) == float("-inf")
    ]
    if not unbounded:
        return dag, True

    # Every starting edge is required. Only additions are legal, and an addition
    # changes one child. With multiple unbounded families every neighbour ties.
    if len(unbounded) > 1:
        return dag, False
    child = unbounded[0]
    parents = list(dag.predecessors(child))
    candidates = set(nodes) - {child} - set(parents) - nx.descendants(dag, child)
    best_parent = None
    best_score = float("-inf")
    for parent in sorted(candidates):
        score = scorer.local_score(child, parents + [parent])
        if score > best_score:
            best_parent, best_score = parent, score
    if best_parent is None:
        return dag, False
    dag.add_edge(best_parent, child)
    return dag, True


# Change done: pass the candidate family scorer and required rule edges to pgmpy.
def run_interval_bic_hillclimb(df_samples, full_table, scoring, *, family_scores=None):
    """
    Run greedy hill climbing using interval BIC (low/mid/high)
    df_samples: LCN forward samples
    full_table: LCN Aggregate/Contingency table
    """

    df = df_samples.astype("category")
    scorer = IntervalBICScore(df, full_table, scoring=scoring, family_scores=family_scores)
    hc = HillClimbSearch(df)
    if family_scores is None:
        best_model = hc.estimate(scoring_method=scorer)
    else:
        knowledge = ExpertKnowledge(required_edges=family_scores.required_edges)
        start, can_search = _rq1_start_dag(df.columns, scorer, family_scores.required_edges)
        if not can_search:
            return start.edges(), float("-inf")
        best_model = hc.estimate(scoring_method=scorer, start_dag=start, expert_knowledge=knowledge)

    # Compute total score using the DAG's parent sets
    total_score = sum(
        scorer.local_score(node, list(best_model.predecessors(node))) 
        for node in best_model.nodes()
    )

    return best_model.edges(), total_score



# Interval BIC scorer for pgmpy
class IntervalBICScore(StructureScore):
    """
    HillClimb scoring using interval BIC for a given aggregate contingency table.

    Supports 'low', 'mid', 'high' collapsing strategies.
    """

    def __init__(self, data, full_table, scoring="mid", *, family_scores=None):
        super().__init__(data)
        self.family_scores = family_scores
        self.full_table = full_table
        self.scoring = scoring

    def local_score(self, node, parents):
        """
        Compute the node-level score using interval BIC for the candidate parent set.
        """
        # Change done: score the candidate's actual parent set for each RQ1 search edit.
        if self.family_scores is None:
            # Compatibility with callers supplying only the legacy aggregate table.
            df_regrouped = regroup_for_parents(self.full_table, node, parents)
            BIC_lower, BIC_upper = compute_interval_BIC(df_regrouped)[node]
        else:
            BIC_lower, BIC_upper = self.family_scores.interval(node, parents)

        # Collapse interval
        if self.scoring == "mid":
            return float((BIC_lower + BIC_upper) / 2)
        elif self.scoring == "low":
            return float(BIC_lower)
        elif self.scoring == "high":
            return float(BIC_upper)
        else:
            raise ValueError("scoring must be 'mid', 'low', or 'high'")