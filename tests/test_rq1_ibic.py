"""Regression checks for the binary RQ1 derivation (run with unittest)."""

import math
import unittest

import pandas as pd

from scoring_functions.updated_ibic import RQ1FamilyScores, _empirical_bounds, binary_loglikelihood_bounds
from structure_learning.hill_climbing import run_interval_bic_hillclimb


# Change done: verify the derivation's empirical BIC, exact endpoints and rule-based dimensions.
class TestRQ1IBIC(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            [(0, 0)] * 4 + [(0, 1), (1, 0)] + [(1, 1)] * 4, columns=["X", "Y"]
        )

    def test_empirical_singletons_root_addition_and_reversal(self):
        scores = RQ1FamilyScores(self.data, [], interval_width=0)
        root = 10 * math.log(0.5) - math.log(10) / 2
        conditional = 8 * math.log(0.8) + 2 * math.log(0.2) - math.log(10)
        for left, right in (("X", "Y"), ("Y", "X")):
            for value in scores.interval(left, ()):
                self.assertAlmostEqual(value, root)
            for value in scores.interval(right, (left,)):
                self.assertAlmostEqual(value, conditional)
                self.assertAlmostEqual(value - root, 0.77615502372055)

    # Change done: cover width propagation across counts, clipping, and unobserved rows.
    def test_propagated_full_width(self):
        for k, n in [(10, 25), (40, 100)]:
            lo, hi = _empirical_bounds(k, n, 0.2)
            self.assertAlmostEqual(lo, 0.3)
            self.assertAlmostEqual(hi, 0.5)
        self.assertEqual(_empirical_bounds(0, 0, 0.4), (0.3, 0.7))
        self.assertEqual(_empirical_bounds(0, 5, 0.4), (0.0, 0.2))
        self.assertEqual(_empirical_bounds(5, 5, 0.4), (0.8, 1.0))
        for width in (-0.1, 1.1, math.nan):
            with self.assertRaises(ValueError):
                RQ1FamilyScores(self.data, [], interval_width=width)

    # Change done: an infinite-score tie must never become an improving pgmpy move.
    def test_unbounded_start(self):
        data = pd.DataFrame([(0, 0)] * 80 + [(1, 0)] * 16 + [(1, 1)] * 4, columns=["A", "Y"])
        for rank in ("low", "mid"):
            fit = RQ1FamilyScores(data, [], interval_width=0.2)
            edges, score = run_interval_bic_hillclimb(data, None, rank, family_scores=fit)
            self.assertEqual(set(edges), {("A", "Y")})
            self.assertTrue(math.isfinite(score))
            tied = data.assign(A=data["Y"])
            fit = RQ1FamilyScores(tied, [], interval_width=0.2)
            edges, score = run_interval_bic_hillclimb(tied, None, rank, family_scores=fit)
            self.assertEqual(set(edges), set())
            self.assertEqual(score, -math.inf)

    def test_row_extrema_at_boundary_and_interior(self):
        f = lambda p: math.log1p(-p) + 4 * math.log(p)
        for lo, hi, best in [(0.2, 0.7, 0.7), (0.3, 0.9, 0.8), (0.9, 0.95, 0.9)]:
            low, high = binary_loglikelihood_bounds(4, 1, lo, hi)
            self.assertAlmostEqual(low, min(f(lo), f(hi)))
            self.assertAlmostEqual(high, f(best))
        self.assertEqual(binary_loglikelihood_bounds(0, 5, 0, 0), (0, 0))
        self.assertEqual(binary_loglikelihood_bounds(1, 5, 0, 0), (-math.inf, -math.inf))
        self.assertEqual(binary_loglikelihood_bounds(0, 0, 0, 1), (0, 0))

    def test_rules_fix_rows_and_only_impossible_configurations_lose_dimension(self):
        data = pd.DataFrame({"X": [0, 0, 1, 1], "Y": [0, 0, 1, 1], "Z": [0, 1, 0, 1]})
        rule = {"if": {"X": True}, "then": {"Y": True}}
        fit = RQ1FamilyScores(data, [rule], interval_width=0)
        for score in fit.interval("Y", ["X"]):
            self.assertAlmostEqual(score, -math.log(4) / 2)
        # (X=0,Y=1) is possible but unobserved. (X=1,Y=0) is impossible.
        for score in fit.interval("Z", ["X", "Y"]):
            self.assertAlmostEqual(score, 4 * math.log(0.5) - 3 * math.log(4) / 2)
        with self.assertRaises(ValueError):
            fit.interval("Y", [])

    def test_supplied_cpt_uses_both_state_bounds_and_observed_counts(self):
        lcn = {"edges": [], "credal_sets": {"X": {"[]": {"True": [0.1, 0.7],
                                                          "False": [0.4, 0.8]}}}}
        fit = RQ1FamilyScores(self.data[["X"]], [], generating_lcn=lcn)
        low, high = fit.interval("X", [])
        penalty = math.log(10) / 2
        self.assertAlmostEqual(low, 5 * math.log(0.2 * 0.8) - penalty)
        self.assertAlmostEqual(high, 10 * math.log(0.5) - penalty)
