import json
import os
from itertools import product

import numpy as np
import pandas as pd
import random

from workflows.app_lcn_gen import generate_basic_lcn, load_dataset, optimize_lcn_bic, optimize_lcn_ibic
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def generate_initial_lcn_graph(csv_data):
    """
    Generate an initial graph structure which will then be 'filled in' 
    Input is the csv data path
    """
    initial_graph, df = generate_basic_lcn(csv_data)

    return initial_graph,df


def run_lcn_bic_optimization(initial_lcn, df):
    """
    - Use GHC + mutations (add/remove/reverse edges)
    - Score each candidate structure using BIC
    - Keep improving until convergence -> final LCN

    - Requires originally loaded csv data df as well
    """

    bic_lcn = optimize_lcn_bic(initial_lcn=initial_lcn, df=df)

    return bic_lcn



def run_lcn_ibic_optimization(initial_lcn, df):
    """
    - Use same search procedure and mutations as BIC version
    - Use Interval BIC (IBIC) instead of BIC
    - Output LCN
    """

    ibic_lcn = optimize_lcn_ibic(initial_lcn=initial_lcn, df=df)

    return ibic_lcn



def _parse_parents(edge_list, node):
    """Deterministic parent ordering"""
    return sorted([u for (u, v) in edge_list if v == node])


def _extract_prob(triple, mode="mid"):
    low, high = float(triple[0]), float(triple[1])

    if mode == "low":
        return low
    elif mode == "high":
        return high
    else:
        return (low + high) / 2


def _normalize(p_true, p_false):
    total = p_true + p_false
    if total == 0:
        return 0.5, 0.5
    return p_true / total, p_false / total


def _get_configurations(parents):
    """
    Produces stable binary ordering for CPT columns:
    e.g. 00, 01, 10, 11
    """
    return list(product([0, 1], repeat=len(parents)))


def _format_config_key(parents, config):
    """
    Converts binary tuple (0,1,1) into credal key format:
    [X1=False, X2=True, X3=True]
    """
    parts = []
    for p, val in zip(parents, config):
        parts.append(f"{p}={'True' if val == 1 else 'False'}")
    return f"[{', '.join(parts)}]"



def _build_bn(lcn, mode="mid"):
    """
    Builds a PGMPY ready bayesian network
    """

    edges = lcn["edges"]
    credal = lcn["credal_sets"]

    model = BayesianNetwork()
    model.add_nodes_from(lcn["nodes"])
    model.add_edges_from(edges)

    cpds = []

    for node in lcn["nodes"]:

        parents = _parse_parents(edges, node)
        node_credal = credal[node]

        # NO PARENTS CASE
        if "[]" in node_credal:

            dist = node_credal["[]"]

            p_true = _extract_prob(dist["True"], mode)
            p_false = _extract_prob(dist["False"], mode)

            p_true, p_false = _normalize(p_true, p_false)

            cpd = TabularCPD(
                variable=node,
                variable_card=2,
                values=[[p_false], [p_true]]
            )

            cpds.append(cpd)
            continue

        # PARENTS CASE
        evidence = parents
        evidence_card = [2] * len(parents)

        true_row = []
        false_row = []

        for config in _get_configurations(parents):

            key = _format_config_key(parents, config)

            if key not in node_credal:
                # fallback if missing config
                p_true, p_false = 0.5, 0.5
            else:
                dist = node_credal[key]

                p_true = _extract_prob(dist["True"], mode)
                p_false = _extract_prob(dist["False"], mode)

                p_true, p_false = _normalize(p_true, p_false)

            true_row.append(p_true)
            false_row.append(p_false)

        cpd = TabularCPD(
            variable=node,
            variable_card=2,
            values=[false_row, true_row],
            evidence=evidence if evidence else None,
            evidence_card=evidence_card if evidence else None
        )

        cpds.append(cpd)

    model.add_cpds(*cpds)

    # strict validation (better error visibility)
    if not model.check_model():
        raise ValueError(f"Invalid BN constructed in mode={mode}")

    return model


def _extract_prob_random(triple):
    low, high = float(triple[0]), float(triple[1])
    return random.uniform(low, high)


def _build_bn_fully_random(lcn, seed=None):
    """
    Random BN baseline:
    - same structure
    - completely random CPTs (ignores credal sets)
    """

    if seed is not None:
        random.seed(seed)

    edges = lcn["edges"]

    model = BayesianNetwork()
    model.add_nodes_from(lcn["nodes"])
    model.add_edges_from(edges)

    cpds = []

    for node in lcn["nodes"]:

        parents = _parse_parents(edges, node)

        # NO PARENTS
        if not parents:
            p_true = random.random()
            p_false = 1 - p_true

            cpd = TabularCPD(
                variable=node,
                variable_card=2,
                values=[[p_false], [p_true]]
            )
            cpds.append(cpd)
            continue

        # WITH PARENTS
        evidence = parents
        evidence_card = [2] * len(parents)

        true_row = []
        false_row = []

        for _ in _get_configurations(parents):

            p_true = random.random()
            p_false = 1 - p_true

            true_row.append(p_true)
            false_row.append(p_false)

        cpd = TabularCPD(
            variable=node,
            variable_card=2,
            values=[false_row, true_row],
            evidence=evidence,
            evidence_card=evidence_card
        )

        cpds.append(cpd)

    model.add_cpds(*cpds)

    if not model.check_model():
        raise ValueError("Invalid random BN")

    return model


def convert_lcn_to_bn(lcn):
    """
    Converts LCN into 3 Bayesian Networks:
    low, mid, high (based on credal interval)
    """

    low_bn = _build_bn(lcn, mode="low")
    mid_bn = _build_bn(lcn, mode="mid")
    high_bn = _build_bn(lcn, mode="high")

    return low_bn, mid_bn, high_bn


def apply_graph_data_prediction(bn, df, target="X14"):
    """
    Apply a Bayesian Network to dataset and evaluate predictive performance.

    Inputs:
    - bn: pgmpy BayesianNetwork (already constructed with CPDs)
    - df: pandas DataFrame (binary data)
    - target: node to predict

    Returns:
    - dict with predictions + evaluation metrics
    """

    infer = VariableElimination(bn)

    y_true = []
    y_pred = []

    nodes = list(df.columns)

    for _, row in df.iterrows():

        # ground truth
        true_value = int(row[target])

        # evidence = all other variables
        evidence = {
            col: int(row[col])
            for col in nodes
            if col != target
        }

        try:
            result = infer.query(
                variables=[target],
                evidence=evidence,
                show_progress=False
            )

            # binary classification: pick most likely state
            pred_value = int(np.argmax(result.values))

        except Exception:
            # fallback if inference fails
            pred_value = 0

        y_true.append(true_value)
        y_pred.append(pred_value)

    # metrics
    output = {
        "target": target,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n_samples": len(y_true),
        "predictions": y_pred[:20],  # sample preview
        "truth": y_true[:20]
    }

    return output


def save_lcn_to_json(lcn, path):
    """
    Save LCN structure to JSON for reuse.
    """

    # ensure folder exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(lcn, f, indent=4)


def load_lcn_from_json(path):
    with open(path, "r") as f:
        return json.load(f)


def run_predictions(bns, targets, df):
    """
    Run predictions for multiple Bayesian Networks and targets.
    Each result is saved individually as:

    prediction_results/{bn_name}_{target}.json
    """

    base_dir = "prediction_results"
    os.makedirs(base_dir, exist_ok=True)

    results = {}

    for bn_name, bn in bns.items():
        results[bn_name] = {}

        for target in targets:

            print(f"Running prediction: {bn_name} -> {target}")

            res = apply_graph_data_prediction(
                bn=bn,
                df=df,
                target=target
            )

            # store in memory structure
            results[bn_name][target] = res

            # required file save
            file_path = os.path.join(
                base_dir,
                f"{bn_name}_{target}.json"
            )

            with open(file_path, "w") as f:
                json.dump(res, f, indent=4)

            print("Saved result")

    return results



def run_application_workflow(csv_data):

    """
    Run 1: initial graph -> GHC + mutations -> BIC scoring -> LCN1
    Run 2: same initial graph -> GHC + mutations -> IBIC scoring -> LCN2

    Note: Overall point is to generate a graph and improve it using different approachs
    Then test predication on the data and see which optimisation approach worked best i.e
    using the IBIC vs BIC

    Note: Most steps have been commented out due to using saved json versions
    """
    

    # Step 1: Load data and generate initial LCN graph 
    # Df is a dataframe of the csv data 
    # initial_lcn, df = generate_initial_lcn_graph(csv_data)
    df = load_dataset(csv_data)

    # Step 2: Optimize initial graph using BIC
    # bic_lcn = optimize_lcn_bic(initial_lcn, df)
    # save_lcn_to_json(bic_lcn, "med_lcns/med_bic_lcn.json")
    bic_lcn = load_lcn_from_json("med_lcns/med_bic_lcn.json")

    # Step 3: Optimize initial graph using IBIC
    # ibic_lcn = run_lcn_ibic_optimization(initial_lcn, df)
    # save_lcn_to_json(ibic_lcn, "med_lcns/med_ibic_lcn.json")
    ibic_lcn = load_lcn_from_json("med_lcns/med_ibic_lcn.json")


    # Step 4: Extract a Bayesian Networks from the lcn
    # BIC optimised LCN case
    bic_low_bn, bic_mid_bn, bic_high_bn = convert_lcn_to_bn(bic_lcn)

    # IBIC optimised LCN case
    ibic_low_bn, ibic_mid_bn, ibic_high_bn = convert_lcn_to_bn(ibic_lcn)

    # Step 5: Apply LCNs to the data to make predictions 
    bic_bn_low_predict = apply_graph_data_prediction(bic_low_bn, df, target="X14")

    # print("Results: ")
    print(bic_bn_low_predict)


    # Random BN 
    bn_random = _build_bn_fully_random(ibic_lcn)

    # Step 5:
    bns = {
        "bic_low": bic_low_bn,
        "bic_mid": bic_mid_bn,
        "bic_high": bic_high_bn,
        "ibic_low": ibic_low_bn,
        "ibic_mid": ibic_mid_bn,
        "ibic_high": ibic_high_bn,
        "bn_rand": bn_random
    }

    targets = ["X7", "X8", "X9"]

    results = run_predictions(
        bns=bns,
        targets=targets,
        df=df
    )

    print("Completed all predictions.")

    return 


def summarise_application_results(
    results_dir="prediction_results",
    output_csv="prediction_summary.csv"
):
    """
    Reads all prediction result JSON files and summarises into a CSV.

    Output format:
    model, target, accuracy, precision, recall, f1
    """

    rows = []

    for file in os.listdir(results_dir):

        if not file.endswith(".json"):
            continue

        file_path = os.path.join(results_dir, file)

        # 🔹 Extract model + target from filename
        # e.g. "bic_low_X14.json" → model="bic_low", target="X14"
        name = file.replace(".json", "")
        parts = name.split("_")

        # model name = everything except last part
        model = "_".join(parts[:-1])
        target_from_filename = parts[-1]

        # Load JSON
        with open(file_path, "r") as f:
            data = json.load(f)

        # Extract metrics (use JSON target as ground truth)
        row = {
            "model": model,
            "target": data.get("target", target_from_filename),
            "accuracy": data.get("accuracy"),
            "precision": data.get("precision"),
            "recall": data.get("recall"),
            "f1": data.get("f1"),
            "n_samples": data.get("n_samples")
        }

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Optional: sort nicely
    df = df.sort_values(by=["model", "target"]).reset_index(drop=True)

    # Save CSV
    df.to_csv(output_csv, index=False)

    print(f"Saved summary to: {output_csv}")
    print(df.head())

    return df