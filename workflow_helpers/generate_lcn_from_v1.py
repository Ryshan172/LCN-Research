import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Backup version of med data LCN generator 

"""
Standalone CSV -> LCN generator.

Input:
    A CSV containing binary variables encoded as 0/1, e.g.
    X1,X2,X3
    0,1,0
    1,0,1

Output:
    A dictionary in the LCN form used by the RQ2 experiment:
        {
            "nodes": [...],
            "edges": [[parent, child], ...],
            "credal_sets": {...},
            "logical_constraints": [...]
        }

The generator:
1. Reads binary CSV data.
2. Builds an initial DAG using pairwise absolute correlation.
3. Estimates conditional probabilities from the data.
4. Converts those probabilities into credal intervals.
5. Adds Boolean implication constraints for deterministic conditional
   relationships, e.g.
       {"if": {"X2": False}, "then": {"X3": True}}
       {"if": {"X2": True},  "then": {"X3": True}}

No pgmpy or other LCN-generation modules are required.
"""


# ---------------------------------------------------------------------
# Semantic feature definitions
# ---------------------------------------------------------------------
#
# These map the anonymised LCN variables back to the features used
# during dataset construction.
#
# Keep this separate from the CSV so the final dataset remains
# anonymised.

FEATURE_METADATA = {
    "X1": {
        "name": "age_elderly",
        "description": "Patient age >= 65",
    },
    "X2": {
        "name": "age_middle",
        "description": "Patient age >= 40 and < 65",
    },
    "X3": {
        "name": "age_young",
        "description": "Patient age < 40",
    },
    "X4": {
        "name": "male",
        "description": "Patient gender is male",
    },
    "X5": {
        "name": "los_short",
        "description": "Length of stay < 48 hours",
    },
    "X6": {
        "name": "los_medium",
        "description": "Length of stay >= 48 and < 120 hours",
    },
    "X7": {
        "name": "los_long",
        "description": "Length of stay >= 120 hours",
    },
    "X8": {
        "name": "dx_low",
        "description": "Number of diagnoses < 10",
    },
    "X9": {
        "name": "dx_high",
        "description": "Number of diagnoses >= 10",
    },
    "X10": {
        "name": "emergency",
        "description": "Admission type contains EMER",
    },
    "X11": {
        "name": "urgent",
        "description": "Admission type contains URGENT",
    },
    "X12": {
        "name": "diabetes",
        "description": "Diabetes indicator",
    },
    "X13": {
        "name": "sepsis",
        "description": "Sepsis indicator",
    },
    "X14": {
        "name": "cardio",
        "description": "Cardiovascular condition indicator",
    },
}


# ---------------------------------------------------------------------
# Semantically defensible logical constraints
# ---------------------------------------------------------------------
#
# These are expressed using the meaningful feature names rather than
# X1, X2, etc.
#
# Each rule means:
#
#     IF all conditions in "if" are true
#     THEN all conditions in "then" must be true.
#
# Boolean values are used exactly as they will be represented in the
# final LCN.
#
SEMANTIC_CONSTRAINTS = [

    # ---------------------------------------------------------------
    # Age categories are mutually exclusive
    # ---------------------------------------------------------------
    #
    # Elderly -> not middle
    {
        "description": "Elderly and middle-aged categories are mutually exclusive",
        "if": {"age_elderly": True},
        "then": {"age_middle": False},
    },

    # Elderly -> not young
    {
        "description": "Elderly and young categories are mutually exclusive",
        "if": {"age_elderly": True},
        "then": {"age_young": False},
    },

    # Middle -> not elderly
    {
        "description": "Middle-aged and elderly categories are mutually exclusive",
        "if": {"age_middle": True},
        "then": {"age_elderly": False},
    },

    # Middle -> not young
    {
        "description": "Middle-aged and young categories are mutually exclusive",
        "if": {"age_middle": True},
        "then": {"age_young": False},
    },

    # Young -> not elderly
    {
        "description": "Young and elderly categories are mutually exclusive",
        "if": {"age_young": True},
        "then": {"age_elderly": False},
    },

    # Young -> not middle
    {
        "description": "Young and middle-aged categories are mutually exclusive",
        "if": {"age_young": True},
        "then": {"age_middle": False},
    },


    # ---------------------------------------------------------------
    # Length-of-stay categories are mutually exclusive
    # ---------------------------------------------------------------
    {
        "description": "Short and medium length-of-stay categories are mutually exclusive",
        "if": {"los_short": True},
        "then": {"los_medium": False},
    },

    {
        "description": "Short and long length-of-stay categories are mutually exclusive",
        "if": {"los_short": True},
        "then": {"los_long": False},
    },

    {
        "description": "Medium and short length-of-stay categories are mutually exclusive",
        "if": {"los_medium": True},
        "then": {"los_short": False},
    },

    {
        "description": "Medium and long length-of-stay categories are mutually exclusive",
        "if": {"los_medium": True},
        "then": {"los_long": False},
    },

    {
        "description": "Long and short length-of-stay categories are mutually exclusive",
        "if": {"los_long": True},
        "then": {"los_short": False},
    },

    {
        "description": "Long and medium length-of-stay categories are mutually exclusive",
        "if": {"los_long": True},
        "then": {"los_medium": False},
    },


    # ---------------------------------------------------------------
    # Diagnosis burden categories are mutually exclusive
    # ---------------------------------------------------------------
    {
        "description": "Low and high diagnosis burden are mutually exclusive",
        "if": {"dx_low": True},
        "then": {"dx_high": False},
    },

    {
        "description": "High and low diagnosis burden are mutually exclusive",
        "if": {"dx_high": True},
        "then": {"dx_low": False},
    },
]


def load_binary_csv(csv_path):
    """Load and validate a CSV containing binary 0/1 variables."""
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    if df.empty:
        raise ValueError("The CSV contains no rows.")

    if len(df.columns) == 0:
        raise ValueError("The CSV contains no columns.")

    for col in df.columns:
        values = set(df[col].dropna().unique())
        if not values.issubset({0, 1, False, True}):
            raise ValueError(
                f"Column '{col}' is not binary. "
                f"Found values: {sorted(values, key=str)}"
            )

    if df.isna().any().any():
        raise ValueError("Missing values are not supported.")

    return df.astype(int)


def generate_structure(df, corr_threshold=0.15, max_parents=2):
    """
    Generate an initial DAG from pairwise absolute correlations.

    The direction follows the CSV column order, matching the original
    app_lcn_gen.py approach: for columns i < j, an accepted relationship
    is represented as (column_i, column_j).

    max_parents limits the number of parents a child can receive.
    """
    if max_parents < 0:
        raise ValueError("max_parents must be >= 0.")

    nodes = list(df.columns)
    edges = []
    parent_count = {node: 0 for node in nodes}

    for i, parent in enumerate(nodes):
        for child in nodes[i + 1:]:
            x = df[parent].to_numpy()
            y = df[child].to_numpy()

            # Constant columns have undefined correlation.
            if np.std(x) == 0 or np.std(y) == 0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(x, y)[0, 1])

            if np.isfinite(corr) and abs(corr) > corr_threshold:
                if parent_count[child] < max_parents:
                    edges.append([parent, child])
                    parent_count[child] += 1

    return edges


def _parent_map(nodes, edges):
    """Build a sorted parent map from the edge list."""
    parents = {node: [] for node in nodes}

    for parent, child in edges:
        if child not in parents:
            raise ValueError(
                f"Edge ({parent}, {child}) contains an unknown child."
            )
        if parent not in parents:
            raise ValueError(
                f"Edge ({parent}, {child}) contains an unknown parent."
            )
        parents[child].append(parent)

    for node in parents:
        parents[node] = sorted(parents[node])

    return parents


def _configuration_key(parents, combo):
    """Create the exact parent-configuration key used by the LCN."""
    return "[" + ", ".join(
        f"{parent}={'True' if value else 'False'}"
        for parent, value in zip(parents, combo)
    ) + "]"


def _probability_interval(p, epsilon):
    """
    Create a valid binary credal interval.

    The False interval is the exact complement of the True interval:
        P(False) = 1 - P(True)
    """
    lower = max(0.0, float(p) - epsilon)
    upper = min(1.0, float(p) + epsilon)

    return {
        "True": [round(lower, 6), round(upper, 6)],
        "False": [round(1.0 - upper, 6), round(1.0 - lower, 6)],
    }


def generate_credal_sets(
    df,
    nodes,
    edges,
    eps_scale=0.15,
    min_eps=0.02,
    deterministic_threshold=1.0,
    deterministic_min_count=1,
):
    """
    Estimate empirical conditional probabilities and create credal sets.

    For a parent configuration with no observations, a weak [0,1]
    probability interval is used.

    For observed configurations, epsilon decreases with sample size.

    If the empirical conditional probability is >= deterministic_threshold,
    the child is treated as deterministically True.

    If it is <= (1 - deterministic_threshold), the child is treated as
    deterministically False.

    With the default deterministic_threshold=1.0, logical constraints are
    generated only for exact deterministic relationships in the CSV.
    """
    if not 0.0 < deterministic_threshold <= 1.0:
        raise ValueError("deterministic_threshold must be in (0, 1].")
    if deterministic_min_count < 1:
        raise ValueError("deterministic_min_count must be >= 1.")
    if eps_scale < 0 or min_eps < 0:
        raise ValueError("eps_scale and min_eps must be >= 0.")

    parent_map = _parent_map(nodes, edges)
    credal_sets = {}
    logical_constraints = []

    for node in nodes:
        parents = parent_map[node]
        credal_sets[node] = {}

        # Root node.
        if not parents:
            p = float(df[node].mean())

            if len(df) < deterministic_min_count:
                raise ValueError(
                    "deterministic_min_count is larger than the dataset."
                )

            if p >= deterministic_threshold:
                interval = {"True": [1.0, 1.0], "False": [0.0, 0.0]}
            elif p <= 1.0 - deterministic_threshold:
                interval = {"True": [0.0, 0.0], "False": [1.0, 1.0]}
            else:
                eps = max(min_eps, eps_scale / np.sqrt(len(df)))
                interval = _probability_interval(p, eps)

            credal_sets[node]["[]"] = interval

            # A root constraint has no parent condition, so it is not
            # represented as an "if"/"then" rule. The requested logical
            # constraint format is conditional, so only non-root nodes
            # generate logical constraints.
            continue

        # Every parent configuration is represented, even if absent in data.
        for combo in itertools.product([False, True], repeat=len(parents)):
            key = _configuration_key(parents, combo)

            mask = np.ones(len(df), dtype=bool)
            for parent, value in zip(parents, combo):
                mask &= df[parent].to_numpy() == int(value)

            subset = df.loc[mask, node]
            n = len(subset)

            if n == 0:
                # No evidence for this configuration.
                interval = {
                    "True": [0.0, 1.0],
                    "False": [0.0, 1.0],
                }
                credal_sets[node][key] = interval
                continue

            p = float(subset.mean())

            # Exact deterministic relationships get exact [0,0]/[1,1]
            # intervals, matching the supplied example.
            if n >= deterministic_min_count and p >= deterministic_threshold:
                interval = {"True": [1.0, 1.0], "False": [0.0, 0.0]}

                logical_constraints.append({
                    "if": {
                        parent: bool(value)
                        for parent, value in zip(parents, combo)
                    },
                    "then": {node: True},
                })

            elif n >= deterministic_min_count and (
                p <= 1.0 - deterministic_threshold
            ):
                interval = {"True": [0.0, 0.0], "False": [1.0, 1.0]}

                logical_constraints.append({
                    "if": {
                        parent: bool(value)
                        for parent, value in zip(parents, combo)
                    },
                    "then": {node: False},
                })

            else:
                eps = max(min_eps, eps_scale / np.sqrt(n))
                interval = _probability_interval(p, eps)

            credal_sets[node][key] = interval

    return credal_sets, logical_constraints


def generate_credal_sets_deterministic(
    df,
    nodes,
    edges,
    eps_scale=0.15,
    min_eps=0.02,
    deterministic_threshold=1.0,
    deterministic_min_count=1,
):
    """
    Estimate empirical conditional probabilities and create credal sets.

    Logical constraints are NOT inferred from empirical probabilities.
    They should instead be supplied separately through
    generate_semantic_constraints().

    Returns
    -------
    credal_sets
        Dictionary containing the credal sets.

    logical_constraints
        Empty list, retained for compatibility with the existing
        calling code.
    """

    if not 0.0 < deterministic_threshold <= 1.0:
        raise ValueError(
            "deterministic_threshold must be in (0, 1]."
        )

    if deterministic_min_count < 1:
        raise ValueError(
            "deterministic_min_count must be >= 1."
        )

    if eps_scale < 0 or min_eps < 0:
        raise ValueError(
            "eps_scale and min_eps must be >= 0."
        )

    parent_map = _parent_map(nodes, edges)

    credal_sets = {}

    # IMPORTANT:
    # Constraints are no longer inferred from the data.
    logical_constraints = []

    for node in nodes:

        parents = parent_map[node]
        credal_sets[node] = {}

        # ---------------------------------------------------------
        # Root node
        # ---------------------------------------------------------
        if not parents:

            p = float(df[node].mean())

            if len(df) < deterministic_min_count:
                raise ValueError(
                    "deterministic_min_count is larger than "
                    "the dataset."
                )

            if p >= deterministic_threshold:

                interval = {
                    "True": [1.0, 1.0],
                    "False": [0.0, 0.0],
                }

            elif p <= 1.0 - deterministic_threshold:

                interval = {
                    "True": [0.0, 0.0],
                    "False": [1.0, 1.0],
                }

            else:

                eps = max(
                    min_eps,
                    eps_scale / np.sqrt(len(df))
                )

                interval = _probability_interval(
                    p,
                    eps
                )

            credal_sets[node]["[]"] = interval

            continue

        # ---------------------------------------------------------
        # Non-root node
        # ---------------------------------------------------------
        for combo in itertools.product(
            [False, True],
            repeat=len(parents)
        ):

            key = _configuration_key(
                parents,
                combo
            )

            mask = np.ones(
                len(df),
                dtype=bool
            )

            for parent, value in zip(
                parents,
                combo
            ):
                mask &= (
                    df[parent].to_numpy()
                    == int(value)
                )

            subset = df.loc[
                mask,
                node
            ]

            n = len(subset)

            # -----------------------------------------------------
            # No observations for this configuration
            # -----------------------------------------------------
            if n == 0:

                interval = {
                    "True": [0.0, 1.0],
                    "False": [0.0, 1.0],
                }

                credal_sets[node][key] = interval

                continue

            p = float(subset.mean())

            # -----------------------------------------------------
            # Deterministic empirical relationship
            #
            # This affects the probability interval only.
            # It does NOT create a logical constraint.
            # -----------------------------------------------------
            if (
                n >= deterministic_min_count
                and p >= deterministic_threshold
            ):

                interval = {
                    "True": [1.0, 1.0],
                    "False": [0.0, 0.0],
                }

            elif (
                n >= deterministic_min_count
                and p <= 1.0 - deterministic_threshold
            ):

                interval = {
                    "True": [0.0, 0.0],
                    "False": [1.0, 1.0],
                }

            else:

                eps = max(
                    min_eps,
                    eps_scale / np.sqrt(n)
                )

                interval = _probability_interval(
                    p,
                    eps
                )

            credal_sets[node][key] = interval

    return credal_sets, logical_constraints

# ---------------------------------------------------------------------
# Semantic constraint generation
# ---------------------------------------------------------------------

def generate_semantic_constraints(
    nodes,
    feature_metadata,
    semantic_constraints,
):
    """
    Convert semantic logical constraints into the anonymised X1/X2/...
    representation expected by the LCN.

    Parameters
    ----------
    nodes : list[str]
        Nodes present in the dataset, e.g. ["X1", "X2", ..., "X14"].

    feature_metadata : dict
        Mapping from anonymised column names to semantic names.

        Example:
            {
                "X1": {"name": "age_elderly"},
                "X2": {"name": "age_middle"},
            }

    semantic_constraints : list[dict]
        Constraints expressed using semantic feature names.

        Example:
            {
                "description": "Elderly implies not middle-aged",
                "if": {"age_elderly": True},
                "then": {"age_middle": False}
            }

    Returns
    -------
    list[dict]
        Constraints in the exact format expected by the LCN.
    """

    # -------------------------------------------------------------
    # Build semantic name -> anonymised column mapping
    # -------------------------------------------------------------
    semantic_to_column = {}

    for column, metadata in feature_metadata.items():
        semantic_name = metadata["name"]

        if semantic_name in semantic_to_column:
            raise ValueError(
                f"Duplicate semantic feature name: '{semantic_name}'"
            )

        semantic_to_column[semantic_name] = column

    # -------------------------------------------------------------
    # Make sure metadata matches the actual dataset
    # -------------------------------------------------------------
    unknown_metadata_nodes = set(feature_metadata) - set(nodes)

    if unknown_metadata_nodes:
        raise ValueError(
            "Feature metadata contains columns not present in the CSV: "
            f"{sorted(unknown_metadata_nodes)}"
        )

    # -------------------------------------------------------------
    # Translate semantic constraints
    # -------------------------------------------------------------
    logical_constraints = []

    for rule in semantic_constraints:

        if "if" not in rule or "then" not in rule:
            raise ValueError(
                "Each semantic constraint must contain 'if' and 'then'."
            )

        semantic_if = rule["if"]
        semantic_then = rule["then"]

        # ---------------------------------------------------------
        # Translate IF side
        # ---------------------------------------------------------
        anonymised_if = {}

        for semantic_name, value in semantic_if.items():

            if semantic_name not in semantic_to_column:
                raise ValueError(
                    f"Unknown semantic feature in constraint: "
                    f"'{semantic_name}'"
                )

            anonymised_if[
                semantic_to_column[semantic_name]
            ] = bool(value)

        # ---------------------------------------------------------
        # Translate THEN side
        # ---------------------------------------------------------
        anonymised_then = {}

        for semantic_name, value in semantic_then.items():

            if semantic_name not in semantic_to_column:
                raise ValueError(
                    f"Unknown semantic feature in constraint: "
                    f"'{semantic_name}'"
                )

            anonymised_then[
                semantic_to_column[semantic_name]
            ] = bool(value)

        # ---------------------------------------------------------
        # Validate that all referenced nodes exist in the LCN
        # ---------------------------------------------------------
        referenced_nodes = (
            set(anonymised_if.keys())
            | set(anonymised_then.keys())
        )

        unknown_nodes = referenced_nodes - set(nodes)

        if unknown_nodes:
            raise ValueError(
                f"Constraint references nodes not present in LCN: "
                f"{sorted(unknown_nodes)}"
            )

        # ---------------------------------------------------------
        # Add only the LCN-compatible structure.
        #
        # "description" is deliberately not included because your
        # current LCN format expects only if/then.
        # ---------------------------------------------------------
        logical_constraints.append({
            "if": anonymised_if,
            "then": anonymised_then,
        })

    return logical_constraints


# TODO: Variations in size / incoming edges etc. 
def generate_lcn_from_csv(
    csv_path,
    corr_threshold=0.15,
    max_parents=2,
    eps_scale=0.15,
    min_eps=0.02,
    deterministic_threshold=1.0,
    deterministic_min_count=1,
    feature_metadata=None,
    semantic_constraints=None,
):
    """
    Main standalone CSV -> LCN function.

    Returns:
        lcn, df

    lcn has the exact top-level structure expected by the RQ2 workflow:
        nodes
        edges
        credal_sets
        logical_constraints
    """
    df = load_binary_csv(csv_path)
    nodes = list(df.columns)

    edges = generate_structure(
        df,
        corr_threshold=corr_threshold,
        max_parents=max_parents,
    )

    # credal_sets, logical_constraints = generate_credal_sets(
    #     df=df,
    #     nodes=nodes,
    #     edges=edges,
    #     eps_scale=eps_scale,
    #     min_eps=min_eps,
    #     deterministic_threshold=deterministic_threshold,
    #     deterministic_min_count=deterministic_min_count,
    # )

    credal_sets, _ = generate_credal_sets_deterministic(
        df=df,
        nodes=nodes,
        edges=edges,
        eps_scale=eps_scale,
        min_eps=min_eps,
        deterministic_threshold=deterministic_threshold,
        deterministic_min_count=deterministic_min_count,
    )

    # -------------------------------------------------------------
    # Generate logical constraints from semantic/domain knowledge
    # -------------------------------------------------------------
    if feature_metadata is not None and semantic_constraints is not None:

        logical_constraints = generate_semantic_constraints(
            nodes=nodes,
            feature_metadata=feature_metadata,
            semantic_constraints=semantic_constraints,
        )

    else:

        logical_constraints = []

    lcn = {
        "nodes": nodes,
        "edges": edges,
        "credal_sets": credal_sets,
        "logical_constraints": logical_constraints,
    }

    return lcn, df


def save_lcn(lcn, output_path):
    """Save the generated LCN as JSON."""
    output_path = Path(output_path)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(lcn, f, indent=2)


def generate_and_save_lcn(
    csv_path,
    output_path="generated_lcn.json",
    corr_threshold=0.15,
    max_parents=2,
    eps_scale=0.15,
    min_eps=0.02,
    deterministic_threshold=1.0,
    deterministic_min_count=1,
):
    """
    Generate an LCN from a binary CSV file and save it as JSON.

    Example:
        lcn = generate_and_save_lcn("data.csv")

    Or with custom settings:
        lcn = generate_and_save_lcn(
            "data.csv",
            "my_lcn.json",
            corr_threshold=0.2,
            max_parents=3,
        )

    Returns:
        dict: The generated LCN.
    """

    df = load_binary_csv(csv_path)
    nodes = list(df.columns)

    edges = generate_structure(
        df,
        corr_threshold=corr_threshold,
        max_parents=max_parents,
    )

    credal_sets, logical_constraints = generate_credal_sets(
        df=df,
        nodes=nodes,
        edges=edges,
        eps_scale=eps_scale,
        min_eps=min_eps,
        deterministic_threshold=deterministic_threshold,
        deterministic_min_count=deterministic_min_count,
    )

    lcn = {
        "nodes": nodes,
        "edges": edges,
        "credal_sets": credal_sets,
        "logical_constraints": logical_constraints,
    }

    save_lcn(lcn, output_path)

    return lcn

