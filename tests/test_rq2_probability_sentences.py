"""Analytic checks for the probability-sentence metrics reported by RQ2."""

import unittest

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from metric_functions.rq2_probability_sentences import evaluate_probability_sentences


class TestProbabilitySentences(unittest.TestCase):
    def test_rules_and_intervals_use_fitted_joint(self):
        # Changed: The fitted graph contains A -> B but still violates A => B.
        model = DiscreteBayesianNetwork([("A", "B")])
        model.add_cpds(TabularCPD("A", 2, [[0.5], [0.5]]),
                       TabularCPD("B", 2, [[0.8, 0.4], [0.2, 0.6]],
                                  evidence=["A"], evidence_card=[2]))
        rules = [{"if": {"A": True}, "then": {"B": True}},
                 {"if": {"B": False}, "then": {"A": False, "B": False}}]
        # Query A | B although B is a child of A in the fitted DAG.
        reference = {"logical_constraints": rules, "credal_sets": {
            "A": {"[B=True]": {"True": [0.4, 0.6], "False": [0.4, 0.6]}},
            "B": {"[]": {"True": [0.4, 0.4], "False": [0.6, 0.6]}}}}
        result = evaluate_probability_sentences(model, reference)["summary"]
        self.assertAlmostEqual(result["any_rule_violation_probability"], 0.2)
        self.assertEqual(result["hard_rule_satisfaction_rate"], 0)
        self.assertEqual(result["interval_row_satisfaction_rate"], 0.5)
        self.assertAlmostEqual(result["max_interval_violation"], 0.15)

    def test_unobserved_states_zero_support_and_empty_inputs(self):
        model = DiscreteBayesianNetwork()
        model.add_node("A")
        model.add_cpds(TabularCPD("A", 1, [[1]], state_names={"A": [True]}))
        model.add_node("B")
        model.add_cpds(TabularCPD("B", 1, [[1]], state_names={"B": [False]}))
        rule = {"if": {"A": False}, "then": {"B": True}}
        bounds = {"True": [1, 1], "False": [0, 0]}
        reference = {"logical_constraints": [rule], "credal_sets": {"B": {"[A=False]": bounds}}}
        result = evaluate_probability_sentences(model, reference)["summary"]
        self.assertEqual(result["any_rule_violation_probability"], 0)
        self.assertEqual(result["hard_rule_zero_support"], 1)
        self.assertEqual(result["interval_row_zero_support"], 1)
        self.assertIsNone(result["interval_row_satisfaction_rate"])
        self.assertIsNone(result["hard_rule_satisfaction_rate"])
        self.assertIsNone(result["max_interval_violation"])

        reference["logical_constraints"] = [{"if": {"A": True}, "then": {"B": False}}]
        self.assertEqual(evaluate_probability_sentences(model, reference)
                         ["summary"]["hard_rule_satisfaction_rate"], 1)
        empty = evaluate_probability_sentences(model, {"logical_constraints": [], "credal_sets": {}})
        self.assertEqual(empty["summary"]["hard_rule_count"], 0)
        self.assertIsNone(empty["summary"]["hard_rule_satisfaction_rate"])
