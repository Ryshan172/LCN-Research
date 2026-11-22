"""Regression check for Boolean state order in the precise baseline BN."""

import unittest
from unittest.mock import patch

from sampler_functions.bn_topological import build_precise_bn_from_lcn


class TestBaselineStateOrder(unittest.TestCase):
    # Change done: check the encoded CPTs and the world sampled by the builder.
    def test_root_and_conditional_probabilities_respect_state_labels(self):
        lcn = {
            "nodes": ["X1", "X5"],
            "edges": [("X1", "X5")],
            "credal_sets": {
                "X1": {"[]": {"True": [0.25, 0.25], "False": [0.75, 0.75]}},
                "X5": {
                    "[X1=False]": {"True": [0, 0], "False": [1, 1]},
                    "[X1=True]": {"True": [0.8, 0.8], "False": [0.2, 0.2]},
                },
            },
            "logical_constraints": [{"if": {"X1": False}, "then": {"X5": False}}],
        }
        for root_draw, expected in [(0.5, False), (0.1, True)]:
            with self.subTest(root_draw=root_draw), patch(
                "sampler_functions.bn_topological.random.random", side_effect=[root_draw, 0.5]
            ):
                model, sampled = build_precise_bn_from_lcn(lcn)
                self.assertTrue(model.check_model())
                self.assertEqual(model.get_cpds("X1").get_value(X1=1), 0.25)
                child = model.get_cpds("X5")
                self.assertEqual(child.get_value(X1=0, X5=0), 1.0)
                self.assertEqual(child.get_value(X1=0, X5=1), 0.0)
                self.assertEqual(child.get_value(X1=1, X5=1), 0.8)
                self.assertEqual(sampled, {"X1": expected, "X5": expected})
