import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lcn_functions.lcn_check import validate_generated_lcn

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


def generate_structure(
    df,
    corr_threshold=0.15,
    in_degree=1,
):
    """
    Generate an initial DAG from pairwise absolute correlations.

    The direction follows the CSV column order:
        for columns i < j, an accepted relationship is
        represented as (column_i, column_j).

    Parameters
    ----------
    df : pandas.DataFrame
        Binary CSV data.

    corr_threshold : float
        Minimum absolute correlation required for an edge.

    in_degree : int
        Maximum number of parents a child may receive.

    Returns
    -------
    list
        List of [parent, child] edges.
    """
    if in_degree < 0:
        raise ValueError("in_degree must be >= 0.")

    nodes = list(df.columns)
    edges = []

    parent_count = {
        node: 0
        for node in nodes
    }

    for i, parent in enumerate(nodes):

        for child in nodes[i + 1:]:

            x = df[parent].to_numpy()
            y = df[child].to_numpy()

            # Constant columns have undefined correlation.
            if np.std(x) == 0 or np.std(y) == 0:
                corr = 0.0
            else:
                corr = float(
                    np.corrcoef(x, y)[0, 1]
                )

            if (
                np.isfinite(corr)
                and abs(corr) > corr_threshold
            ):
                if parent_count[child] < in_degree:

                    edges.append([
                        parent,
                        child,
                    ])

                    parent_count[child] += 1

    return edges


def _generate_random_width(
    interval_width,
    dist_type="beta",
):
    """
    Generate a random interval width using the same distribution
    choices as the standard LCN generator.

    Parameters
    ----------
    interval_width : float
        Maximum/nominal width of the generated probability interval.

    dist_type : str
        Distribution used to generate the width.

        Supported:
            "beta"
            "gaussian"
            "uniform"
            "triangular"

    Returns
    -------
    float
        Generated interval width.
    """
    if interval_width < 0:
        raise ValueError(
            "interval_width must be >= 0."
        )

    dist_type = dist_type.lower()

    if dist_type == "beta":
        # Beta(2, 2) produces values concentrated around 0.5.
        width = np.random.beta(2, 2)

    elif dist_type == "gaussian":
        width = np.random.normal(
            loc=0.5,
            scale=0.15,
        )

    elif dist_type == "uniform":
        width = np.random.uniform(
            low=0.0,
            high=1.0,
        )

    elif dist_type == "triangular":
        width = np.random.triangular(
            left=0.0,
            mode=0.5,
            right=1.0,
        )

    else:
        raise ValueError(
            f"Unsupported width distribution: '{dist_type}'. "
            "Use 'beta', 'gaussian', 'uniform', or 'triangular'."
        )

    # Keep the sampled value within [0, 1].
    width = float(
        np.clip(width, 0.0, 1.0)
    )

    # Scale by requested interval width.
    width *= interval_width

    return float(
        np.clip(width, 0.0, 1.0)
    )


def _probability_interval_random(
    p,
    interval_width,
    width_dist_type="beta",
):
    """
    Convert an empirical probability into a credal interval using
    a randomly generated interval width.

    The False interval is the exact complement of the True interval.
    """
    p = float(p)

    width = _generate_random_width(
        interval_width=interval_width,
        dist_type=width_dist_type,
    )

    lower = max(
        0.0,
        p - width / 2.0,
    )

    upper = min(
        1.0,
        p + width / 2.0,
    )

    return {
        "True": [
            round(lower, 6),
            round(upper, 6),
        ],
        "False": [
            round(1.0 - upper, 6),
            round(1.0 - lower, 6),
        ],
    }


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
    interval_width=0.2,
    width_dist_type="beta",
    deterministic_threshold=1.0,
    deterministic_min_count=1,
):
    """
    Estimate empirical conditional probabilities from the CSV and
    convert them into credal intervals.

    interval_width and width_dist_type control the uncertainty
    around the empirical probabilities.

    Logical constraints are NOT inferred from the empirical data.
    They are supplied separately through generate_semantic_constraints().

    Parameters
    ----------
    df : pandas.DataFrame
        Binary CSV data.

    nodes : list[str]
        LCN nodes.

    edges : list
        LCN edges.

    interval_width : float
        Width parameter used for credal intervals.

    width_dist_type : str
        Distribution used to sample interval widths.

    deterministic_threshold : float
        Probability threshold for exact deterministic intervals.

    deterministic_min_count : int
        Minimum number of observations required before treating a
        relationship as deterministic.

    Returns
    -------
    credal_sets : dict
        Generated credal sets.

    logical_constraints : list
        Empty list. Semantic constraints are generated separately.
    """

    if interval_width < 0:
        raise ValueError(
            "interval_width must be >= 0."
        )

    if not 0.0 < deterministic_threshold <= 1.0:
        raise ValueError(
            "deterministic_threshold must be in (0, 1]."
        )

    if deterministic_min_count < 1:
        raise ValueError(
            "deterministic_min_count must be >= 1."
        )

    # Validate the distribution before generation.
    _generate_random_width(
        interval_width=interval_width,
        dist_type=width_dist_type,
    )

    parent_map = _parent_map(
        nodes,
        edges,
    )

    credal_sets = {}

    # Semantic/domain constraints are handled separately.
    logical_constraints = []

    for node in nodes:

        parents = parent_map[node]

        credal_sets[node] = {}

        # ---------------------------------------------------------
        # Root node
        # ---------------------------------------------------------
        if not parents:

            p = float(
                df[node].mean()
            )

            if len(df) < deterministic_min_count:
                raise ValueError(
                    "deterministic_min_count is larger than "
                    "the dataset."
                )

            # Exact deterministic root.
            if p >= deterministic_threshold:

                interval = {
                    "True": [1.0, 1.0],
                    "False": [0.0, 0.0],
                }

            # Exact deterministic false root.
            elif p <= (
                1.0 - deterministic_threshold
            ):

                interval = {
                    "True": [0.0, 0.0],
                    "False": [1.0, 1.0],
                }

            else:

                interval = _probability_interval_random(
                    p=p,
                    interval_width=interval_width,
                    width_dist_type=width_dist_type,
                )

            credal_sets[node]["[]"] = interval

            continue

        # ---------------------------------------------------------
        # Non-root node
        # ---------------------------------------------------------
        for combo in itertools.product(
            [False, True],
            repeat=len(parents),
        ):

            key = _configuration_key(
                parents,
                combo,
            )

            mask = np.ones(
                len(df),
                dtype=bool,
            )

            for parent, value in zip(
                parents,
                combo,
            ):

                mask &= (
                    df[parent].to_numpy()
                    == int(value)
                )

            subset = df.loc[
                mask,
                node,
            ]

            n = len(subset)

            # -----------------------------------------------------
            # No observations for this configuration
            # -----------------------------------------------------
            if n == 0:

                credal_sets[node][key] = {
                    "True": [0.0, 1.0],
                    "False": [0.0, 1.0],
                }

                continue

            p = float(
                subset.mean()
            )

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
                and p <= (
                    1.0 - deterministic_threshold
                )
            ):

                interval = {
                    "True": [0.0, 0.0],
                    "False": [1.0, 1.0],
                }

            else:

                interval = _probability_interval_random(
                    p=p,
                    interval_width=interval_width,
                    width_dist_type=width_dist_type,
                )

            credal_sets[node][key] = interval

    return (
        credal_sets,
        logical_constraints,
    )



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


def _select_lcn_columns(
    df,
    size,
):
    """
    Select exactly `size` variables from the CSV to form the LCN.

    The first `size` CSV columns are selected, preserving their original
    order.

    Parameters
    ----------
    df : pandas.DataFrame
        Full binary CSV dataset.

    size : int
        Desired number of LCN nodes.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing exactly `size` selected columns.
    """

    if not isinstance(size, (int, np.integer)):
        raise ValueError(
            f"size must be an integer. Got: {type(size).__name__}"
        )

    size = int(size)

    if size < 1:
        raise ValueError(
            "size must be >= 1."
        )

    available_size = len(df.columns)

    if size > available_size:
        raise ValueError(
            f"Requested LCN size ({size}) is larger than the "
            f"number of variables in the CSV ({available_size})."
        )

    selected_columns = list(
        df.columns[:size]
    )

    return df.loc[
        :,
        selected_columns
    ].copy()


def _filter_semantic_constraints_for_nodes(
    nodes,
    feature_metadata,
    semantic_constraints,
):
    """
    Keep only semantic constraints whose referenced features are
    present in the generated LCN.

    Constraints involving variables excluded by the requested LCN
    size are skipped.

    This is important when, for example, the CSV contains X1-X14
    but size=10.
    """

    if (
        feature_metadata is None
        or semantic_constraints is None
    ):
        return []

    selected_nodes = set(nodes)

    selected_semantic_names = {
        feature_metadata[node]["name"]
        for node in selected_nodes
        if node in feature_metadata
    }

    filtered_constraints = []

    for rule in semantic_constraints:

        if "if" not in rule or "then" not in rule:
            raise ValueError(
                "Each semantic constraint must contain 'if' and 'then'."
            )

        referenced_features = (
            set(rule["if"].keys())
            | set(rule["then"].keys())
        )

        # ---------------------------------------------------------
        # If the rule refers to a feature that isn't part of the
        # requested LCN, it cannot be represented in this LCN.
        # ---------------------------------------------------------
        if not referenced_features.issubset(
            selected_semantic_names
        ):
            continue

        filtered_constraints.append(
            rule
        )

    return filtered_constraints


# Separates one generation attempt from the retry logic.
def _generate_csv_candidate(
    df,
    size,
    interval_width,
    width_dist_type,
    in_degree,
    corr_threshold=0.15,
    feature_metadata=None,
    semantic_constraints=None,
    deterministic_threshold=1.0,
    deterministic_min_count=1,
):
    """
    Generate one candidate LCN containing exactly `size` nodes.

    The full CSV may contain more variables than the requested LCN
    size. Only the first `size` columns are used.
    """

    # -------------------------------------------------------------
    # Select exactly `size` variables.
    # -------------------------------------------------------------
    lcn_df = _select_lcn_columns(
        df=df,
        size=size,
    )

    nodes = list(
        lcn_df.columns
    )

    # -------------------------------------------------------------
    # Generate structure using ONLY the selected variables.
    # -------------------------------------------------------------
    edges = generate_structure(
        df=lcn_df,
        corr_threshold=corr_threshold,
        in_degree=in_degree,
    )

    # -------------------------------------------------------------
    # Generate credal sets using ONLY the selected variables.
    # -------------------------------------------------------------
    credal_sets, _ = generate_credal_sets_deterministic(
        df=lcn_df,
        nodes=nodes,
        edges=edges,
        interval_width=interval_width,
        width_dist_type=width_dist_type,
        deterministic_threshold=deterministic_threshold,
        deterministic_min_count=deterministic_min_count,
    )

    # -------------------------------------------------------------
    # Filter semantic constraints so that only constraints whose
    # variables exist in this particular LCN size are retained.
    # -------------------------------------------------------------
    filtered_constraints = (
        _filter_semantic_constraints_for_nodes(
            nodes=nodes,
            feature_metadata=feature_metadata,
            semantic_constraints=semantic_constraints,
        )
    )

    # -------------------------------------------------------------
    # Generate semantic/domain constraints.
    # -------------------------------------------------------------
    if (
        feature_metadata is not None
        and filtered_constraints
    ):

        logical_constraints = (
            generate_semantic_constraints(
                nodes=nodes,
                feature_metadata=feature_metadata,
                semantic_constraints=filtered_constraints,
            )
        )

    else:

        logical_constraints = []

    # -------------------------------------------------------------
    # Assemble candidate LCN.
    # -------------------------------------------------------------
    lcn = {
        "nodes": nodes,
        "edges": edges,
        "credal_sets": credal_sets,
        "logical_constraints": logical_constraints,
    }

    # -------------------------------------------------------------
    # Final safety check.
    # -------------------------------------------------------------
    if len(lcn["nodes"]) != size:
        raise RuntimeError(
            f"Internal generation error: requested size={size}, "
            f"but generated {len(lcn['nodes'])} nodes."
        )

    return lcn


def generate_lcn_from_csv(
    csv_path,
    size,
    interval_width=0.2,
    width_dist_type="beta",
    in_degree=1,
    corr_threshold=0.15,
    feature_metadata=None,
    semantic_constraints=None,
    deterministic_threshold=1.0,
    deterministic_min_count=1,
    max_attempts=10,
):
    """
    Generate a valid LCN from a binary CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to the binary CSV file.

    size : int
        Desired number of nodes in the generated LCN.

        If the CSV contains more variables than `size`, only the
        first `size` variables are used.

        Example:
            CSV = X1,...,X14
            size = 5

        produces:

            nodes = ["X1", "X2", "X3", "X4", "X5"]

        If `size` is greater than the number of CSV columns, an
        exception is raised.

    interval_width : float
        Width parameter used when generating credal intervals.

    width_dist_type : str
        Distribution used to generate interval widths.

        Supported:
            "beta"
            "gaussian"
            "uniform"
            "triangular"

    in_degree : int
        Maximum number of parents allowed for each node.

    corr_threshold : float
        Minimum absolute correlation required to create an edge.

    feature_metadata : dict, optional
        Mapping from X1/X2/... to semantic feature names.

    semantic_constraints : list, optional
        Domain/semantic logical constraints.

    deterministic_threshold : float
        Threshold for exact deterministic probability intervals.

    deterministic_min_count : int
        Minimum observations required for deterministic intervals.

    max_attempts : int
        Maximum number of candidate-generation attempts.

    Returns
    -------
    lcn, df
        The valid LCN and the full loaded dataframe.

    Notes
    -----
    `size` directly controls the number of nodes in the generated LCN.

    The CSV is loaded in full, but only the first `size` columns are
    used for LCN generation.
    """

    # -------------------------------------------------------------
    # Validate parameters.
    # -------------------------------------------------------------
    if not isinstance(size, (int, np.integer)):
        raise ValueError(
            f"size must be an integer. Got: {type(size).__name__}"
        )

    size = int(size)

    if size < 1:
        raise ValueError(
            "size must be >= 1."
        )

    if max_attempts < 1:
        raise ValueError(
            "max_attempts must be >= 1."
        )

    if in_degree < 0:
        raise ValueError(
            "in_degree must be >= 0."
        )

    if interval_width < 0:
        raise ValueError(
            "interval_width must be >= 0."
        )

    # -------------------------------------------------------------
    # Load CSV once.
    # -------------------------------------------------------------
    df = load_binary_csv(
        csv_path
    )

    available_size = len(
        df.columns
    )

    if available_size == 0:
        raise ValueError(
            "The CSV contains no variables."
        )

    # -------------------------------------------------------------
    # size now ACTUALLY determines the LCN size.
    # -------------------------------------------------------------
    if size > available_size:
        raise ValueError(
            f"Requested LCN size ({size}) is larger than the "
            f"number of variables in the CSV ({available_size})."
        )

    selected_nodes = list(
        df.columns[:size]
    )

    print(
        f"Generating LCN from CSV: "
        f"requested_size={size}, "
        f"available_variables={available_size}, "
        f"selected_variables={selected_nodes}, "
        f"interval_width={interval_width}, "
        f"width_dist_type='{width_dist_type}', "
        f"in_degree={in_degree}"
    )

    # -------------------------------------------------------------
    # Retry until a valid LCN is generated.
    # -------------------------------------------------------------
    is_valid = False
    attempts = 0

    while (
        not is_valid
        and attempts < max_attempts
    ):

        attempts += 1

        print(
            f"\nGenerating candidate LCN "
            f"(size={size}, "
            f"attempt {attempts}/{max_attempts})..."
        )

        # ---------------------------------------------------------
        # Generate a NEW candidate.
        #
        # Because interval widths are randomly sampled, each
        # attempt can produce a different candidate.
        # ---------------------------------------------------------
        lcn = _generate_csv_candidate(
            df=df,
            size=size,
            interval_width=interval_width,
            width_dist_type=width_dist_type,
            in_degree=in_degree,
            corr_threshold=corr_threshold,
            feature_metadata=feature_metadata,
            semantic_constraints=semantic_constraints,
            deterministic_threshold=deterministic_threshold,
            deterministic_min_count=deterministic_min_count,
        )

        # ---------------------------------------------------------
        # Validate candidate using the same validation function
        # used by the normal LCN generation workflow.
        # ---------------------------------------------------------
        is_valid = validate_generated_lcn(
            lcn
        )

        print(
            f"Attempt {attempts}: "
            f"Is Valid? {is_valid}"
        )

    # -------------------------------------------------------------
    # Failed after maximum attempts.
    # -------------------------------------------------------------
    if not is_valid:
        raise RuntimeError(
            f"Failed to generate valid LCN of size={size} "
            f"after {max_attempts} attempts."
        )

    # -------------------------------------------------------------
    # Final node-count assertion.
    # -------------------------------------------------------------
    if len(lcn["nodes"]) != size:
        raise RuntimeError(
            f"Generated LCN has {len(lcn['nodes'])} nodes, "
            f"but requested size={size}."
        )

    print(
        f"\nValid LCN of size={size} "
        f"generated on attempt {attempts}."
    )

    return lcn, df


def save_lcn(lcn, output_path):
    """Save the generated LCN as JSON."""
    output_path = Path(output_path)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(lcn, f, indent=2)


def generate_and_save_lcn(
    csv_path,
    output_path="generated_lcn.json",
    size=None,
    interval_width=0.2,
    width_dist_type="beta",
    in_degree=1,
    corr_threshold=0.15,
    feature_metadata=FEATURE_METADATA,
    semantic_constraints=SEMANTIC_CONSTRAINTS,
    deterministic_threshold=1.0,
    deterministic_min_count=1,
    max_attempts=10,
):
    """
    Generate a valid LCN from a binary CSV file and save it as JSON.

    Parameters
    ----------
    csv_path : str or Path
        Input binary CSV.

    output_path : str or Path
        Output JSON path.

    size : int, optional
        Desired number of nodes in the generated LCN.

        If None, all variables in the CSV are used.

        If supplied and smaller than the number of CSV columns,
        only the first `size` columns are used.

        If supplied and larger than the number of CSV columns,
        ValueError is raised.

    interval_width : float
        Credal interval width parameter.

    width_dist_type : str
        Distribution used for interval widths.

        Supported:
            "beta"
            "gaussian"
            "uniform"
            "triangular"

    in_degree : int
        Maximum number of incoming edges per node.

    corr_threshold : float
        Correlation threshold used for structure generation.

    feature_metadata : dict, optional
        Semantic metadata.

    semantic_constraints : list, optional
        Semantic/domain constraints.

    deterministic_threshold : float
        Threshold for deterministic probability intervals.

    deterministic_min_count : int
        Minimum observations required for deterministic intervals.

    max_attempts : int
        Maximum number of generation/validation attempts.

    Returns
    -------
    dict
        The valid generated LCN.
    """

    # -------------------------------------------------------------
    # Backwards-compatible behaviour:
    # if size is omitted, use every variable in the CSV.
    # -------------------------------------------------------------
    if size is None:

        df = load_binary_csv(
            csv_path
        )

        size = len(
            df.columns
        )

        if size == 0:
            raise ValueError(
                "The CSV contains no variables."
            )

    # -------------------------------------------------------------
    # Generate the LCN.
    # -------------------------------------------------------------
    lcn, _ = generate_lcn_from_csv(
        csv_path=csv_path,
        size=size,
        interval_width=interval_width,
        width_dist_type=width_dist_type,
        in_degree=in_degree,
        corr_threshold=corr_threshold,
        feature_metadata=feature_metadata,
        semantic_constraints=semantic_constraints,
        deterministic_threshold=deterministic_threshold,
        deterministic_min_count=deterministic_min_count,
        max_attempts=max_attempts,
    )

    # -------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------
    save_lcn(
        lcn,
        output_path,
    )

    return lcn



#TODO: Constraints not being generated like in old file. See why