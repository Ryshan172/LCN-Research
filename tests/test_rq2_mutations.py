import unittest
from unittest.mock import patch

from mutations import mutation_functions as mutation
from workflows import rq2_experiments as rq2
from workflows import rq2_med_app_workflow as med
from workflows import rq3_experiments as rq3


# Changed: Cover feasibility, retries, repair, runner inputs, and shared RQ3 score accounting.
class RQ2MutationTests(unittest.TestCase):
    def setUp(self):
        self.nodes = ["X", "Y", "Z"]
        self.rules = [{"if": {"X": True}, "then": {"Y": True}}]
        self.state = rq2.initialise_structure(self.nodes, self.rules, "repair", 1)
        self.index = self.state["constraint_index"]

    def test_starting_graphs_and_impossible_priors(self):
        self.assertEqual(self.state["edges"], [("X", "Y")])
        self.assertEqual(rq2.initialise_structure(self.nodes, self.rules)["edges"], [])
        for source, target in [("Y", "X"), ("Z", "Y"), ("W", "Z"), ("X", "X")]:
            rules = self.rules + [{"if": {source: True}, "then": {target: True}}]
            with self.subTest(source=source, target=target), self.assertRaises(ValueError):
                rq2.hill_climb(None, None, self.nodes, rules, "constraint", max_in_degree=1)

    def test_standard_retries_duplicate_cycle_and_parent_cap(self):
        for invalid in [("X", "Y"), ("Y", "X"), ("Z", "Y")]:
            with self.subTest(invalid=invalid), patch.object(
                mutation.random, "sample", side_effect=[invalid, ("Y", "Z")]
            ) as draw:
                result = mutation.standard_mutation(self.state, max_attempts=2)
                self.assertEqual(result["edges"], [("X", "Y"), ("Y", "Z")])
                self.assertEqual(draw.call_count, 2)
        self.assertEqual(self.state["edges"], [("X", "Y")])

    def test_rejection_retries_from_incumbent(self):
        self.state["edges"].append(("Y", "Z"))
        with patch.object(mutation.random, "choice", side_effect=self.state["edges"]):
            result = mutation.constraint_aware_mutation(self.state, self.index, "edge_delete")
        self.assertEqual(result["edges"], [("X", "Y")])
        self.assertEqual(len(self.state["edges"]), 2)

    def test_required_deletion_reversal_and_repair_budget(self):
        for edit in ["edge_delete", "edge_flip", "edge_reverse"]:
            for budget in [0, 1, 3]:
                with self.subTest(edit=edit, budget=budget):
                    result = mutation.post_mutation_contraint_repair(
                        self.state, self.index, edit, max_repairs=budget, edge=("X", "Y")
                    )
                    self.assertIs(result, self.state)
        result = mutation.post_mutation_contraint_repair(
            self.state, self.index, "edge_add", max_repairs=0, edge=("Y", "Z")
        )
        self.assertEqual(result["edges"], [("X", "Y"), ("Y", "Z")])

    def test_cycle_outside_prior_never_returned(self):
        self.state["edges"].append(("Y", "Z"))
        with patch.object(mutation.random, "sample", return_value=("Z", "X")):
            for strategy in ["standard", "constraint", "repair"]:
                self.assertIs(rq2.mutate_lcn(self.state, strategy), self.state)

    def test_noop_skips_scoring_but_records_iterations(self):
        for strategy in ["standard", "constraint", "repair"]:
            with self.subTest(strategy=strategy), patch.object(
                rq2, "score_structure", return_value=1
            ) as score:
                best, trajectory = rq2.hill_climb(
                    None, None, self.nodes, self.rules, strategy, "edge_reverse",
                    max_iter=3, max_in_degree=1
                )
                self.assertEqual(score.call_count, 1)
                self.assertEqual(trajectory, [1, 1, 1, 1])
                self.assertEqual(best["edges"], [] if strategy == "standard" else [("X", "Y")])

    def test_valid_reversal_and_strict_score_acceptance(self):
        result = mutation.standard_mutation(self.state, "edge_reverse", edge=("X", "Y"))
        self.assertEqual(result["edges"], [("Y", "X")])
        for new_score, accepted in [(2, True), (1, False), (0, False)]:
            with patch.object(rq2, "score_structure", side_effect=[1, new_score]), patch.object(
                mutation.random, "sample", return_value=("Y", "Z")
            ):
                best, _ = rq2.hill_climb(None, None, self.nodes, self.rules,
                                         "repair", max_iter=1, max_in_degree=1)
                self.assertEqual(("Y", "Z") in best["edges"], accepted)

    def test_rq3_targeted_repair_can_start_with_incomplete_prior(self):
        rules = self.rules + [{"if": {"Z": True}, "then": {"W": True}}]
        initial = rq2.initialise_structure(self.nodes + ["W", "V"], rules)
        candidate = mutation.post_mutation_contraint_repair(
            initial, initial["constraint_index"], edge=("Y", "V")
        )
        self.assertEqual(set(candidate["edges"]), {("X", "Y"), ("Y", "V")})

    def test_rq3_counts_score_calls_and_first_best_evaluation(self):
        initial = rq2.initialise_structure(self.nodes)
        first = dict(initial, edges=[("X", "Y")])
        rejected = dict(initial, edges=[("Y", "Z")])
        def edits(current, **kwargs):
            return next(draws) or current

        draws = iter([None, first, None, rejected])
        with patch.object(rq2, "mutate_lcn", side_effect=edits), patch.object(
            rq2, "score_structure", side_effect=[1, 2, 2]
        ) as score:
            best, trajectory, diagnostics = rq3.run_search(
                "rq2_ghc", None, None, initial, "standard", max_evaluations=5, seed=71
            )
        self.assertEqual(best["edges"], first["edges"])
        self.assertEqual(trajectory, [1, 2, 2])
        self.assertEqual(diagnostics["evaluations"], score.call_count)
        self.assertEqual(diagnostics["evaluations_to_best"], 2)

    def test_experiment_runners_pass_dictionary_rules_and_parent_cap(self):
        model = dict(self.state, edges=[("X", "Y")])
        for module, generator, run, extra in [
            (rq2, "generate_lcn", rq2.run_experiment_steps, {}),
            (med, "generate_and_save_lcn", med.run_rq2_med_experiment_steps, {"run_num": 1}),
        ]:
            with self.subTest(module=module.__name__), patch.object(
                module, generator, return_value=model
            ), patch.object(module, "sample_lcn", return_value=(None, None)), patch.object(
                rq2, "score_structure"
            ) as score:
                with self.assertRaisesRegex(ValueError, "No feasible starting DAG"):
                    run(size=3, interval_width=.3, width_dist_type="uniform", in_degree=0,
                        num_samples=10, mutation_strategy="constraint", **extra)
                score.assert_not_called()
