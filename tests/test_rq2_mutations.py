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


class RQ2ExperimentStartTests(unittest.TestCase):
    def test_starts_are_matched_and_preserve_the_fixed_edit(self):
        nodes = ["X", "Y", "Z", "W"]
        rules = [{"if": {"X": True}, "then": {"Y": True}}]
        for edit in ["edge_delete", "edge_reverse", "edge_flip"]:
            for cap in [1, 2]:
                for seed in range(8):
                    with self.subTest(edit=edit, cap=cap, seed=seed):
                        edges, available = rq2.experiment_start(nodes, rules, "constraint", edit, cap, seed)
                        self.assertEqual((edges, available), rq2.experiment_start(nodes, rules, "repair", edit, cap, seed))
                        self.assertTrue(available)
                        self.assertTrue(mutation.valid_dag(nodes, edges, cap))
                        self.assertIn(("X", "Y"), edges)
                        self.assertLessEqual(len(edges) - 1, len(nodes) - 1)
                        state = rq2.initialise_structure(nodes, rules, "repair", cap, initial_edges=edges)
                        changed = mutation.post_mutation_contraint_repair(
                            state, state["constraint_index"], edit, edge=edges[-1]
                        )
                        self.assertIsNot(changed, state)
                        self.assertIn(("X", "Y"), changed["edges"])

    def test_addition_and_baseline_knowledge(self):
        nodes = ["X", "Y", "Z"]
        rules = [{"if": {"X": True}, "then": {"Y": True}}]
        self.assertEqual(rq2.experiment_start(nodes, rules, "repair", "edge_add", 1, 0), ([("X", "Y")], True))
        self.assertEqual(rq2.experiment_start(nodes, rules, "standard", "edge_add", 1, 0), ([], True))
        for edit in ["edge_delete", "edge_reverse"]:
            self.assertEqual(rq2.experiment_start(nodes, rules, "standard", edit, 1, 1),
                             rq2.experiment_start(nodes, [], "standard", edit, 1, 1))

    def test_distinguish_deletable_from_reversible_and_impossible(self):
        nodes = ["X", "Y", "Z"]
        rules = [{"if": {"X": True}, "then": {"Y": True}},
                 {"if": {"Y": True}, "then": {"Z": True}}]
        self.assertTrue(rq2.experiment_start(nodes, rules, "repair", "edge_delete", 2, 12)[1])
        # Changed: X -> Z can be inserted but its reversal would close the required chain.
        edges, available = rq2.experiment_start(nodes, rules, "repair", "edge_reverse", 2, 12)
        self.assertFalse(available)
        self.assertEqual(set(edges), {("X", "Y"), ("Y", "Z")})
        for edit in ["edge_add", "edge_delete", "edge_reverse"]:
            self.assertEqual(rq2.experiment_start(nodes, [], "standard", edit, 0, 12), ([], False))

    def test_conflicting_draw_is_replaced_before_any_strategy_runs(self):
        import pandas as pd
        from unittest.mock import Mock

        first = {"if": {"X1": False}, "then": {"X5": True}}
        second = {"if": {"X2": True}, "then": {"X5": False}}
        nodes = ["X1", "X2", "X5"]
        bad = {"nodes": nodes, "logical_constraints": [first, second]}
        good = {"nodes": nodes, "logical_constraints": [first]}
        sampled = pd.DataFrame([[True, True, False]], columns=nodes)
        make = Mock(side_effect=[bad, good])
        with patch.object(rq2, "sample_lcn", return_value=(sampled, None)):
            trials = list(rq2.paired_trials([3], [2], ["standard", "constraint", "repair"],
                                            ["edge_add"], 1, make, 20))
        self.assertEqual(make.call_count, 2)
        self.assertEqual([row[0] for row in trials], [1, 2, 3])
        self.assertTrue(all(row[-2][0] is good for row in trials))

    def test_grid_shares_data_seed_and_preserves_run_ids(self):
        from unittest.mock import Mock
        make_model = Mock(side_effect=[{"nodes": []}, {"nodes": []}])
        with patch.object(rq2, "sample_lcn", side_effect=[("d1", "a1"), ("d2", "a2")]) as sample:
            trials = list(rq2.paired_trials([5], [1], ["standard", "constraint", "repair"],
                                            ["edge_delete", "edge_reverse"], 2, make_model, 20))
        self.assertEqual([trial[0] for trial in trials], [1, 3, 5, 2, 4, 6])
        self.assertEqual(make_model.call_count, 2)
        self.assertEqual(sample.call_count, 2)
        for group in (trials[:3], trials[3:]):
            for trial in group:
                self.assertIs(trial[-2], group[0][-2])
                self.assertEqual(trial[-1], group[0][-1])


class RQ2IBICParityTests(unittest.TestCase):
    # Changed: Compare the graph scores used by RQ2 with RQ1's search scorer
    # on identical data, including the rule-free baseline and two widths.
    def test_scores_match_rq1_for_each_strategy_and_mode(self):
        import pandas as pd
        from scoring_functions.updated_ibic import IBICScorer
        from structure_learning.hill_climbing import IntervalBICScore

        data = pd.DataFrame([(0, 0), (0, 1), (1, 1), (1, 1)], columns=["X", "Y"])
        rules = [{"if": {"X": True}, "then": {"Y": True}}]
        for strategy in ("standard", "constraint", "repair"):
            state = rq2.initialise_structure(data.columns, rules, strategy, 1,
                                             initial_edges=[("X", "Y")])
            for width in (0, 0.3):
                actual = rq2.rq2_family_scores(data, rules, strategy, width)
                expected = IBICScorer(data, [] if strategy == "standard" else rules,
                                           interval_width=width)
                for mode in ("low", "mid", "high"):
                    with self.subTest(strategy=strategy, width=width, mode=mode):
                        rq1 = IntervalBICScore(data, None, scoring=mode, family_scores=expected)
                        reference = rq1.local_score("X", []) + rq1.local_score("Y", ["X"])
                        self.assertEqual(rq2.score_structure(state, data, None, mode,
                                                             family_scores=actual), reference)

    def test_upper_endpoint_crosses_plateau_before_midpoint_becomes_finite(self):
        import pandas as pd
        pair = [(0, 0)] * 80 + [(1, 0)] * 16 + [(1, 1)] * 4
        data = pd.DataFrame([left + right for left in pair for right in pair],
                            columns=["A", "Y", "B", "Z"])
        families = rq2.rq2_family_scores(data, [], "standard", 0.2)
        for search in (rq2.hill_climb, rq2.tabu_search, rq2.random_restart_hill_climb):
            edits = iter([("A", "Y"), ("B", "Z")])
            def mutate(state, **kwargs):
                return dict(state, edges=state["edges"] + [next(edits)])
            upper = []
            with patch.object(rq2, "mutate_lcn", side_effect=mutate):
                best, mid = search(data, None, list(data), max_iter=2, family_scores=families,
                                   upper_trajectory=upper,
                                   **({"n_restarts": 1} if search == rq2.random_restart_hill_climb else {}))
            self.assertEqual(mid[:2], [float("-inf")] * 2)
            self.assertGreater(upper[1], upper[0])
            self.assertGreater(mid[2], mid[1])
            self.assertEqual(best["edges"], [("A", "Y"), ("B", "Z")])
        self.assertFalse(rq2._improves((1, 5), (1, 2)))
        self.assertFalse(rq2._improves((float("-inf"), 1e-12), (float("-inf"), 0)))

    def test_unbounded_restarts_keep_a_graph_and_save_the_actual_score(self):
        import json
        import pandas as pd

        data = pd.DataFrame({"X": [False] * 19 + [True]})
        families = rq2.rq2_family_scores(data, [], "standard", 0.3)
        graph, trajectory = rq2.random_restart_hill_climb(
            data, None, ["X"], n_restarts=2, max_iter=1, family_scores=families)
        self.assertEqual(graph["edges"], [])
        self.assertEqual(trajectory, [float("-inf")] * 4)
        metrics, saved = rq2.rq2_ibic_results(trajectory)
        json.dumps([metrics, saved], allow_nan=False)
        self.assertEqual(saved, ["-Infinity"] * 4)
        self.assertEqual(metrics["best_score"], "-Infinity")
        self.assertIsNone(metrics["stability"])
