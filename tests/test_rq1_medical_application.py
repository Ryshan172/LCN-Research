"""Regression checks for the medical RQ1 scoring and prediction path."""
import math
import unittest
from unittest.mock import patch

import pandas as pd

from scoring_functions.updated_ibic import RQ1FamilyScores
from workflows.app_lcn_gen import optimize_lcn_ibic
from workflows.application_workflow import convert_lcn_to_bn


class TestMedicalIBIC(unittest.TestCase):
    # Change done: exercise a -inf tie on observed records, then verify that
    # prediction uses the selected family's bounds, including unobserved outcomes.
    def test_midpoint_tie_and_prediction_endpoints(self):
        df = pd.DataFrame({"A": [0] * 96 + [1] * 4, "Y": [0] * 96 + [1] * 4})
        initial = {"nodes": list(df), "edges": [], "logical_constraints": []}
        with patch("workflows.app_lcn_gen.mutate_edges", return_value=[("A", "Y")]):
            result = optimize_lcn_ibic(initial, df, max_iters=1)
        self.assertEqual(result["edges"], [("A", "Y")])
        self.assertEqual(result["ibic_score"], -math.inf)
        fit = RQ1FamilyScores(df, [])
        self.assertEqual(result["credal_sets"]["Y"], fit.fitted_cpt("Y", ["A"]))
        low, mid, high = convert_lcn_to_bn(result, complementary=True)
        for bn, expected in [(low, 0.9), (mid, 0.95), (high, 1.0)]:
            self.assertAlmostEqual(bn.get_cpds("Y").get_value(A=1, Y=1), expected)

    def test_forced_row_and_finite_score(self):
        df = pd.DataFrame({"A": [0, 0, 1, 1], "Y": [0, 0, 0, 1]})
        rule = {"if": {"A": False}, "then": {"Y": False}}
        initial = {"nodes": list(df), "edges": [], "logical_constraints": [rule]}
        result = optimize_lcn_ibic(initial, df, max_iters=0)
        self.assertEqual(result["edges"], [("A", "Y")])
        self.assertEqual(result["credal_sets"]["Y"]["[A=False]"]["True"], [0.0, 0.0])
        fit = RQ1FamilyScores(df, [rule])
        expected = sum(sum(fit.interval(n, parents)) / 2 for n, parents in [("A", []), ("Y", ["A"])])
        self.assertAlmostEqual(result["ibic_score"], expected)
