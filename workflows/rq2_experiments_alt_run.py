"""Paired synthetic RQ2 experiment comparing midpoint IBIC and BIC.

From the project root: code_env/bin/python -m workflows.rq2_experiments_alt_run
Use --help for run controls. Each output directory contains paired observations,
an overall comparison, and results stratified by width, training size and edit.
"""

import argparse
import json
import math
import random
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from pgmpy.estimators import BIC, BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork
from scipy.stats import t

from metric_functions.kl_divergence import log_prob_sample
from mutations.mutation_functions import extract_constraint_dependencies, valid_dag
from sampler_functions.contingency_sampler import sample_dataset
from scoring_functions.updated_ibic import UnscorableRQ1Draw
from workflows import rq2_experiments as rq2


STRATEGIES = ("standard", "constraint", "repair")
EDITS = ("edge_add", "edge_delete", "edge_reverse")
SIZE, PARENT_CAP = 7, 2


class BICFamilies:
    """Adapt ordinary pgmpy BIC to the existing RQ2 family-score interface."""

    def __init__(self, train):
        states = {node: [False, True] for node in train}
        self.bic = BIC(train.astype("category"), state_names=states)
        self.cache = {}

    def interval(self, node, parents):
        # Changed: A point interval lets the same RQ2 search call ordinary BIC.
        key = (node, tuple(sorted(parents)))
        if key not in self.cache:
            value = self.bic.local_score(node, list(key[1]))
            self.cache[key] = (value, value)
        return self.cache[key]


def draw_trial(seed, width, n_records):
    """Generate one feasible model and collect independent records from its sampler."""
    random.seed(seed)
    np.random.seed(seed)
    model = rq2.generate_lcn(SIZE, width, "uniform", PARENT_CAP)
    rules = model["logical_constraints"]
    required = extract_constraint_dependencies(rules)
    if not valid_dag(model["nodes"], required, PARENT_CAP):
        raise UnscorableRQ1Draw("Required arcs do not form a feasible RQ2 starting graph")

    # The RQ2 sampler rejects records that violate a rule. Count accepted
    # records so that a nominal training size means the same in both arms.
    records = []
    for _ in range(20):
        records.extend(sample_dataset(model, n_records - len(records)))
        if len(records) == n_records:
            break
    else:
        raise UnscorableRQ1Draw("Insufficient accepted records after 20 sampling batches")
    return model, pd.DataFrame(records, columns=model["nodes"])


def evaluate_graph(edges, train, test_counts, reference_edges, required):
    """Measure held-out joint log loss and directed edge errors beyond the prior."""
    fitted = DiscreteBayesianNetwork(edges)
    fitted.add_nodes_from(train.columns)
    # Changed: Fit both selected graphs with the same training-only estimator.
    # A BDeu prior of one equivalent record gives unseen test events finite loss.
    fitted.fit(train, estimator=BayesianEstimator, prior_type="BDeu",
               equivalent_sample_size=1, state_names={n: [False, True] for n in train})
    loss = -math.fsum(count * log_prob_sample(fitted, dict(zip(train.columns, world)))
                      for world, count in test_counts.items()) / test_counts.sum()
    # Symmetric difference counts one error per wrong directed arc. Exclude
    # all supplied arcs from both sides, including for the standard baseline.
    errors = len((reference_edges ^ set(edges)) - required)
    return {"test_log_loss": loss, "unknown_arc_errors": errors}


def paired_rows(model, records, train_sizes, width, repeat, seed, draw_attempt, iterations):
    """Share samples, starting graphs and mutation seeds within each comparison."""
    nodes, rules = model["nodes"], model["logical_constraints"]
    required = extract_constraint_dependencies(rules)
    truth = set(map(tuple, model["edges"]))
    # Nested training sets allow sample-size comparisons on the same model.
    # The disjoint test block is shared by every arm and never enters search.
    test_counts = records.iloc[max(train_sizes):].value_counts(sort=False)
    # Give the search a separate random stream from the generating graph.
    streams = random.Random(f"rq2-alt-search-{seed}")
    starts = {edit: (streams.getrandbits(32), streams.getrandbits(32)) for edit in EDITS}
    for n_train in train_sizes:
        train = records.iloc[:n_train]
        bic = BICFamilies(train)
        evaluations = {}
        for strategy in STRATEGIES:
            families = rq2.rq2_family_scores(train, rules, strategy, width)
            for edit in EDITS:
                start_seed, mutation_seed = starts[edit]
                initial, available = rq2.experiment_start(nodes, rules, strategy, edit,
                                                          PARENT_CAP, start_seed)
                arms = [("ibic", families, "mid"), ("ordinary_bic", bic, "mid")]
                if strategy != "standard":
                    # RQ1 centres every free interval on the empirical MLE and
                    # uses the same dimension penalty at both endpoints. Its upper
                    # endpoint is therefore BIC with the very same hard-rule support.
                    arms.append(("rule_matched_bic", families, "high"))
                outcomes = {}
                for name, scorer, mode in arms:
                    # Changed: Call the real RQ2 hill climber in every arm. Reset
                    # its RNG for the pair; accepted moves may later diverge.
                    random.seed(mutation_seed)
                    upper_scores = []
                    best, scores = rq2.hill_climb(
                        train, None, nodes, logical_constraints=rules, mutation_strategy=strategy,
                        mutation_type=edit, max_iter=iterations, scoring=mode,
                        max_in_degree=PARENT_CAP, record_noops=False,
                        initial_edges=initial, family_scores=scorer, upper_trajectory=upper_scores)
                    edges = tuple(sorted(map(tuple, best["edges"])))
                    if edges not in evaluations:
                        evaluations[edges] = evaluate_graph(edges, train, test_counts, truth, required)
                    outcomes[name] = {
                        **evaluations[edges], "edges": json.dumps(edges),
                        "upper_score": upper_scores[-1], "search_score": scores[-1], "score_evaluations": len(scores),
                    }
                ibic = outcomes.pop("ibic")
                for comparator, baseline in outcomes.items():
                    yield {
                        "repeat": repeat, "draw_seed": seed, "draw_attempt": draw_attempt,
                        "size": SIZE, "parent_cap": PARENT_CAP, "interval_width": width,
                        "training_rows": n_train, "test_rows": int(test_counts.sum()),
                        "strategy": strategy, "edit": edit, "iterations": iterations,
                        "start_seed": start_seed, "mutation_seed": mutation_seed,
                        "initial_move_available": available, "initial_edges": json.dumps(initial),
                        "reference_edges": json.dumps(sorted(truth)), "required_arcs": json.dumps(sorted(required)),
                        "comparator": comparator, **{"ibic_" + k: v for k, v in ibic.items()},
                        **{"bic_" + k: v for k, v in baseline.items()},
                        # Positive paired gains always favour IBIC. Raw search
                        # objectives are diagnostic and are never subtracted.
                        "log_loss_gain": baseline["test_log_loss"] - ibic["test_log_loss"],
                        "unknown_arc_error_reduction": baseline["unknown_arc_errors"] - ibic["unknown_arc_errors"],
                    }


def summarize(frame, keys):
    """Summarize paired gains; each input group has one observation per repeat."""
    rows = []
    for labels, group in frame.groupby(keys, sort=True):
        gain = group["log_loss_gain"]
        mean = gain.mean()
        # Two-sided, approximate 95% t interval across independent repeats.
        # A single repeat cannot estimate between-repeat variability.
        margin = t.ppf(0.975, len(gain) - 1) * gain.sem() if len(gain) > 1 else np.nan
        ties = gain.abs() <= 1e-10
        rows.append({
            **dict(zip(keys, labels if isinstance(labels, tuple) else (labels,))),
            "repeats": len(gain), "mean_log_loss_gain": mean,
            "ci95_low": mean - margin, "ci95_high": mean + margin,
            "ibic_wins": int((gain > 1e-10).sum()), "ties": int(ties.sum()),
            "bic_wins": int((gain < -1e-10).sum()),
            "mean_unknown_arc_error_reduction": group["unknown_arc_error_reduction"].mean(),
        })
    return pd.DataFrame(rows)


def run_experiment(output_dir, *, repeats=20, iterations=100, seed=1729,
                   train_sizes=(50, 200), widths=(0.1, 0.3), test_size=500):
    """Run the prespecified grid and write results separately from original RQ2."""
    if min(repeats, iterations, test_size) < 1 or not train_sizes or min(train_sizes) < 1:
        raise ValueError("Repeats, iterations and record counts must be positive")
    if not widths or any(not 0 <= w <= 1 for w in widths):
        raise ValueError("Full interval widths must be in [0, 1]")
    if len(set(widths)) != len(widths) or len(set(train_sizes)) != len(train_sizes):
        raise ValueError("Training sizes and widths must each be distinct")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / name for name in ("paired_runs.csv", "by_condition.csv", "overall.csv", "draws.csv")]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Alternate run outputs already exist in {output}; select another directory")
    seeds = random.Random(seed)
    # Sample feasibility before examining any search outcome. Log every draw,
    # including rejected ones, and never retry a weak result or a flat trajectory.
    for repeat, width in product(range(1, repeats + 1), widths):
        for attempt in range(1, 11):
            draw_seed = seeds.getrandbits(32)
            reason = ""
            try:
                model, records = draw_trial(draw_seed, width, max(train_sizes) + test_size)
                # Check rule consistency on training data before running any arm.
                rq2.rq2_family_scores(records.iloc[:max(train_sizes)], model["logical_constraints"],
                                      "constraint", width)
            except UnscorableRQ1Draw as exc:
                reason = str(exc)
            except RuntimeError as exc:
                if not str(exc).startswith("Failed to generate valid LCN"):
                    raise
                reason = str(exc)
            audit = dict(master_seed=seed, repeat=repeat, interval_width=width, attempt=attempt,
                         draw_seed=draw_seed, accepted=not reason, rejection_reason=reason)
            first_draw = repeat == 1 and width == widths[0] and attempt == 1
            pd.DataFrame([audit]).to_csv(paths[3], mode="x" if first_draw else "a",
                                         header=first_draw, index=False)
            if not reason:
                break
        else:
            raise RuntimeError(f"No usable draw for repeat {repeat}, width {width}; see {paths[3]}")
        rows = pd.DataFrame(paired_rows(model, records, train_sizes, width, repeat, draw_seed,
                                        attempt, iterations))
        rows.to_csv(paths[0], mode="a", header=not paths[0].exists(), index=False)

    pairs = pd.read_csv(paths[0])
    summarize(pairs, ["comparator", "interval_width", "training_rows", "strategy", "edit"]).to_csv(
        paths[1], index=False)
    # Strategies, edits and nested samples share a generating model. Average
    # within each repeat first so these correlated runs do not inflate precision.
    per_repeat = pairs.groupby(["comparator", "repeat"], as_index=False)[
        ["log_loss_gain", "unknown_arc_error_reduction"]].mean()
    overall = summarize(per_repeat, ["comparator"])
    overall.to_csv(paths[2], index=False)
    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    output = args.output_dir or Path("results_rq2_alt") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    summary = run_experiment(output, repeats=args.repeats, iterations=args.iterations, seed=args.seed)
    print(summary.to_string(index=False))
    print(f"Saved paired experiment results in {output}")
