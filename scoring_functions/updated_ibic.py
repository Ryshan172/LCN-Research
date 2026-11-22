"""Whole-graph IBIC for the binary, row-specified DAGs used in RQ1."""

from itertools import product
from math import inf, isclose, log, log1p

import networkx as nx
import pandas as pd

from sampler_functions.bn_topological import parse_parent_config_string


# Change done: distinguish a retryable model conflict from other workflow errors.
class UnscorableRQ1Draw(ValueError):
    """A generated draw cannot be scored under the row-wise RQ1 model."""


# Change done: optimise the observed Bernoulli log likelihood over one feasible row.
def binary_loglikelihood_bounds(n_true, n_false, lower, upper):
    """Exact extrema for one Bernoulli row with fixed observation counts."""
    if not 0 <= lower <= upper <= 1:
        raise ValueError(f"Infeasible binary row [{lower}, {upper}]")
    if n_true + n_false == 0:
        return 0.0, 0.0

    def likelihood(p):
        if (n_true and p == 0) or (n_false and p == 1):
            return -inf
        return (n_true * log(p) if n_true else 0.0) + (
            n_false * log1p(-p) if n_false else 0.0
        )

    mle = n_true / (n_true + n_false)
    return min(likelihood(lower), likelihood(upper)), likelihood(min(upper, max(lower, mle)))


# Change done: enforce consistency between the generating CPT's two binary outcomes.
def _binary_bounds(row):
    """Intersect the supplied bounds on True and its complement."""
    lo, hi = row["True"]
    flo, fhi = row["False"]
    if not (0 <= lo <= hi <= 1 and 0 <= flo <= fhi <= 1):
        raise ValueError(f"Invalid CPT bounds: {row}")
    lower, upper = max(lo, 1 - fhi), min(hi, 1 - flo)
    if lower > upper:
        if not isclose(lower, upper, rel_tol=0, abs_tol=1e-15):
            raise UnscorableRQ1Draw(f"No normalized Bernoulli distribution satisfies {row}")
        lower = upper = lo if lo == hi else lower
    return lower, upper


# Change done: propagate the configured full width around each candidate row's own centre.
def _empirical_bounds(n_true, n, interval_width):
    """Use the empirical frequency, or the midpoint for an unobserved possible row."""
    centre = n_true / n if n else 0.5
    radius = interval_width / 2
    return max(0.0, centre - radius), min(1.0, centre + radius)


class RQ1FamilyScores:
    """Score independent CPT rows after enforcing the supplied hard implications.

    Search fits each candidate family from complete records. With a generating
    LCN, the same likelihood is instead evaluated over its supplied CPTs.
    """

    def __init__(self, samples, rules, generating_lcn=None, *, interval_width=0.2):
        if samples.empty or not samples.isin([False, True]).all().all():
            raise ValueError("RQ1 IBIC needs nonempty, complete binary observations")
        if not 0 <= interval_width <= 1:
            raise ValueError("interval_width must be a full probability width in [0, 1]")
        self.interval_width = interval_width
        self.data = samples.astype(bool)
        # Change done: keep implication parents in the search and preserve supplied reference CPTs.
        self.rules = {node: [] for node in samples}
        for rule in rules:
            for child, value in rule["then"].items():
                self.rules[child].append((rule["if"], value))
        self.required_edges = {
            (parent, child) for child, clauses in self.rules.items()
            for conditions, _ in clauses for parent in conditions
        }
        self.parents = {}
        self.cpts = {}
        if generating_lcn is not None:
            for node in samples:
                parents = sorted(p for p, c in generating_lcn["edges"] if c == node)
                self.parents[node] = tuple(parents)
                self.cpts[node] = {}
                for label, bounds in generating_lcn["credal_sets"][node].items():
                    config = parse_parent_config_string(label)
                    if set(config) != set(parents):
                        raise ValueError(f"CPT parent mismatch for {node}: {label}")
                    key = tuple(config[p] for p in parents)
                    self.cpts[node][key] = _binary_bounds(bounds)

        # Change done: determine possible rows from rules and CPT support independently of counts.
        graph = nx.DiGraph()
        graph.add_nodes_from(samples.columns)
        graph.add_edges_from(self.required_edges)
        if generating_lcn is not None:
            graph.add_edges_from(generating_lcn["edges"])
        worlds = [{}]
        for node in nx.topological_sort(graph):
            expanded = []
            for world in worlds:
                lo, hi = 0.0, 1.0
                if self.cpts:
                    key = tuple(world[p] for p in self.parents[node])
                    lo, hi = self.cpts[node][key]
                for conditions, value in self.rules[node]:
                    if self.cpts and not conditions.keys() <= set(self.parents[node]):
                        continue
                    if all(world[p] == v for p, v in conditions.items()):
                        if value:
                            lo = 1.0
                        else:
                            hi = 0.0
                if lo > hi:
                    raise UnscorableRQ1Draw(f"Inconsistent bounds or implications for {node} at {world}")
                for state, possible in ((False, lo < 1), (True, hi > 0)):
                    if possible:
                        expanded.append({**world, node: state})
            worlds = expanded
        if self.cpts:
            for child, clauses in self.rules.items():
                for conditions, value in clauses:
                    if conditions.keys() <= set(self.parents[child]):
                        continue
                    if any(all(w[p] == v for p, v in conditions.items()) and w[child] != value
                           for w in worlds):
                        raise UnscorableRQ1Draw(f"Nonlocal rule for {child} requires joint scoring")
        self.support = pd.DataFrame(worlds, columns=samples.columns)
        self.cache = {}

    def interval(self, node, parents):
        parents = tuple(sorted(parents))
        key = (node, parents)
        if key not in self.cache:
            self.cache[key] = self._family_interval(node, parents)
        return self.cache[key]

    # Change done: export the same fitted rows used for scoring so the medical
    # predictor uses the selected graph's actual feasible probabilities.
    def fitted_cpt(self, node, parents):
        parents = tuple(sorted(parents))
        cpt = {}
        _, upper = self._family_interval(node, parents, cpt)
        if upper == -inf:
            raise ValueError(f"Observed outcomes are impossible for {node}")
        return cpt

    def _family_interval(self, node, parents, cpt=None):
        if self.cpts and parents != self.parents[node]:
            raise ValueError("Generating CPTs only apply to their own parent set")
        if not self.cpts and any(
            p not in parents for conditions, _ in self.rules[node] for p in conditions
        ):
            raise ValueError(f"Candidate family for {node} omits a required rule parent")

        # Change done: count outcomes for the proposed parents and penalise only free, possible rows.
        columns = [*parents, node]
        counts = self.data.value_counts(subset=columns, sort=False).to_dict()
        possible = set(self.support[columns].itertuples(index=False, name=None))
        if not counts.keys() <= possible:
            return -inf, -inf
        low = high = 0.0
        dimension = 0
        for config in product((False, True), repeat=len(parents)):
            false_key, true_key = (*config, False), (*config, True)
            can_false, can_true = false_key in possible, true_key in possible
            key = None
            if cpt is not None:
                key = "[" + ", ".join(f"{p}={v}" for p, v in zip(parents, config)) + "]"
            if not (can_false or can_true):
                # Unreachable rows need a normalized placeholder for pgmpy's complete CPT.
                if cpt is not None:
                    cpt[key] = {"True": [0.5, 0.5], "False": [0.5, 0.5]}
                continue
            n_false, n_true = counts.get(false_key, 0), counts.get(true_key, 0)
            if not (can_false and can_true):
                lo = hi = float(can_true)
            elif self.cpts:
                lo, hi = self.cpts[node][config]
                dimension += int(lo != hi)
            else:
                lo, hi = _empirical_bounds(n_true, n_true + n_false, self.interval_width)
                dimension += 1
            if cpt is not None:
                cpt[key] = {"True": [lo, hi], "False": [1 - hi, 1 - lo]}
            row_low, row_high = binary_loglikelihood_bounds(n_true, n_false, lo, hi)
            low += row_low
            high += row_high
        penalty = 0.5 * dimension * log(len(self.data))
        return low - penalty, high - penalty
