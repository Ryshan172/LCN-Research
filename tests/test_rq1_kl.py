"""Regression tests for exact KL evaluation in synthetic RQ1."""

import math
import unittest

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from metric_functions.kl_divergence import exact_kl_divergence


def root_bn(probabilities, labels):
    model = DiscreteBayesianNetwork()
    model.add_node("X")
    model.add_cpds(TabularCPD("X", len(labels), [[p] for p in probabilities],
                              state_names={"X": labels}))
    return model


# Change done: verify exact KL where RQ1's training-sample estimate was negative.
class TestRQ1KL(unittest.TestCase):
    def test_negative_training_estimate_and_positive_exact_value(self):
        p = root_bn([0.5, 0.5], [0, 1])
        q = root_bn([0.2, 0.8], [0, 1])
        training_average = 0.2 * math.log(0.5 / 0.2) + 0.8 * math.log(0.5 / 0.8)
        self.assertLess(training_average, 0)
        self.assertAlmostEqual(exact_kl_divergence(p, q), math.log(1.25))

    def test_joint_probabilities_node_and_state_order(self):
        p = DiscreteBayesianNetwork([("A", "B")])
        p.add_cpds(TabularCPD("A", 2, [[0.5], [0.5]]),
                   TabularCPD("B", 2, [[0.9, 0.1], [0.1, 0.9]],
                              evidence=["A"], evidence_card=[2]))
        q = DiscreteBayesianNetwork()
        q.add_nodes_from(["B", "A"])
        for node in q.nodes():
            q.add_cpds(TabularCPD(node, 2, [[0.5], [0.5]]))
        self.assertAlmostEqual(exact_kl_divergence(p, q),
                               0.9 * math.log(1.8) + 0.1 * math.log(0.2))
        ordered = root_bn([0.2, 0.8], [0, 1])
        reversed_labels = root_bn([0.8, 0.2], [1, 0])
        self.assertEqual(exact_kl_divergence(ordered, reversed_labels), 0.0)

    def test_zero_probability_and_missing_fitted_state(self):
        p = root_bn([0.5, 0.5], [0, 1])
        for q in (root_bn([1.0, 0.0], [0, 1]), root_bn([1.0], [0])):
            self.assertEqual(exact_kl_divergence(p, q), math.inf)
            self.assertAlmostEqual(exact_kl_divergence(q, p), math.log(2.0))
            self.assertEqual(exact_kl_divergence(q, q), 0.0)
