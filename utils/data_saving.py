import json
import uuid
from pathlib import Path
import numpy as np
import pandas as pd

def make_json_safe(obj):
    """Recursively convert objects to JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": obj.shape,
            "columns": obj.columns.tolist(),
        }
    elif hasattr(obj, "edges"):
        # pgmpy models
        return {
            "type": obj.__class__.__name__,
            "nodes": list(obj.nodes()),
            "edges": list(obj.edges()),
        }
    elif hasattr(obj, "__dict__"):
        return make_json_safe(vars(obj))
    else:
        return obj


def save_experiment_to_json(experiment_res, run_id, output_dir="results"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    file_path = Path(output_dir) / f"{run_id}.json"

    json_safe_res = make_json_safe(experiment_res)

    with open(file_path, "w") as f:
        json.dump(json_safe_res, f, indent=2)

    return str(file_path)
