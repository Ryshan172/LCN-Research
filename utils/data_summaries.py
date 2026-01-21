import json
from pathlib import Path
import pandas as pd

"""
Functions to load a single experiment JSON and summarise it

Read all experiment files from a directory and save CSV
"""


def summarise_experiment_json(json_path):
    """
    Read a single experiment JSON file and return a flat dict
    suitable for CSV storage.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    params = data["params"]
    results = data["results"]

    summary = {
        # ---- parameters ----
        "size": params["size"],
        "interval_width": params["interval_width"],
        "width_dist_type": params["width_dist_type"],
        "in_degree": params["in_degree"],
        "num_samples": params["num_samples"],

        # ---- KL divergences ----
        "kl_baseline_vs_learned_bn": results["kl_divergence"]["baseline_vs_learned_bn"],
        "kl_baseline_vs_lcn_learned_bn": results["kl_divergence"]["baseline_vs_lcn_learned_bn"],

        # ---- baseline SHD ----
        "baseline_hillclimb_shd": results["baseline_structure_learning"]["shd"]["hillclimb_shd"]["shd"],

        # ---- interval BIC scores ----
        "interval_bic_low": results["interval_bic_learning"]["interval_bic_results"]["low"]["score"],
        "interval_bic_mid": results["interval_bic_learning"]["interval_bic_results"]["mid"]["score"],
        "interval_bic_high": results["interval_bic_learning"]["interval_bic_results"]["high"]["score"],

        # ---- interval LCN SHDs ----
        "interval_lcn_shd_low": results["interval_bic_learning"]["interval_lcn_shd"]["low_shd"]["shd"],
        "interval_lcn_shd_mid": results["interval_bic_learning"]["interval_lcn_shd"]["mid_shd"]["shd"],
        "interval_lcn_shd_high": results["interval_bic_learning"]["interval_lcn_shd"]["high_shd"]["shd"],
    }

    return summary


def summarise_experiments_to_csv(input_dir, output_csv):
    """
    Read all JSON experiment files in a directory and
    write a CSV summary.
    """
    summaries = []

    for json_file in Path(input_dir).glob("*.json"):
        try:
            summary = summarise_experiment_json(json_file)
            summary["experiment_file"] = json_file.name
            summaries.append(summary)
        except Exception as e:
            print(f"Skipping {json_file}: {e}")

    df = pd.DataFrame(summaries)
    df.to_csv(output_csv, index=False)

    return df