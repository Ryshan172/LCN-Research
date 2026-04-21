import json
from pathlib import Path
import pandas as pd


def safe_get(dct, path, default=None):
    for key in path:
        if not isinstance(dct, dict) or key not in dct:
            return default
        dct = dct[key]
    return dct


def summarise_experiment_json(json_path):

    with open(json_path, "r") as f:
        data = json.load(f)

    params = data.get("params", {})
    results = data.get("results", {})

    lcn_nodes = safe_get(results, ["lcn", "nodes"], [])
    lcn_edges = safe_get(results, ["lcn", "edges"], [])

    baseline_shd = safe_get(
        results,
        ["baseline_structure_learning", "shd", "hillclimb_shd", "shd"],
        None
    )

    ibic_low = safe_get(results, ["interval_bic_learning", "interval_bic_results", "low", "score"])
    ibic_mid = safe_get(results, ["interval_bic_learning", "interval_bic_results", "mid", "score"])
    ibic_high = safe_get(results, ["interval_bic_learning", "interval_bic_results", "high", "score"])

    shd_low = safe_get(results, ["interval_bic_learning", "interval_lcn_shd", "low_shd", "shd"])
    shd_mid = safe_get(results, ["interval_bic_learning", "interval_lcn_shd", "mid_shd", "shd"])
    shd_high = safe_get(results, ["interval_bic_learning", "interval_lcn_shd", "high_shd", "shd"])

    kl_baseline = safe_get(results, ["kl_divergence", "baseline_vs_learned_bn"])
    kl_lcn = safe_get(results, ["kl_divergence", "baseline_vs_lcn_learned_bn"])

    summary = {
        # run_id will be injected later (NOT from file)
        "repeat": data.get("repeat"),

        "num_samples": params.get("num_samples"),

        # LCN structure
        "lcn_num_nodes": len(lcn_nodes) if lcn_nodes else 0,
        "lcn_num_edges": len(lcn_edges) if lcn_edges else 0,

        # SHD
        "baseline_hillclimb_shd": baseline_shd,

        # IBIC
        "ibic_low": ibic_low,
        "ibic_mid": ibic_mid,
        "ibic_high": ibic_high,

        # interval SHD
        "interval_lcn_shd_low": shd_low,
        "interval_lcn_shd_mid": shd_mid,
        "interval_lcn_shd_high": shd_high,

        # KL
        "kl_baseline_vs_learned_bn": kl_baseline,
        "kl_baseline_vs_lcn_learned_bn": kl_lcn,
    }

    return summary


def summarise_med_experiments(input_dir, output_csv):

    summaries = []

    json_files = sorted(Path(input_dir).glob("*.json"))

    for i, json_file in enumerate(json_files, start=1):
        try:
            summary = summarise_experiment_json(json_file)

            # Set run_id manually
            summary["run_id"] = i

            summary["experiment_file"] = json_file.name
            summaries.append(summary)

        except Exception as e:
            print(f"Skipping {json_file}: {e}")

    df = pd.DataFrame(summaries)
    df.to_csv(output_csv, index=False)

    return df