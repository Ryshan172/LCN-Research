from itertools import product

from workflows.app_lcn_gen import generate_basic_lcn, optimize_lcn_bic, optimize_lcn_ibic
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD


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


def convert_lcn_to_bn(lcn):
    """
    Converts LCN into 3 Bayesian Networks:
    low, mid, high (based on credal interval)
    """

    low_bn = _build_bn(lcn, mode="low")
    mid_bn = _build_bn(lcn, mode="mid")
    high_bn = _build_bn(lcn, mode="high")

    return low_bn, mid_bn, high_bn



def run_application_workflow(csv_data):

    """
    Run 1: initial graph -> GHC + mutations -> BIC scoring -> LCN1
    Run 2: same initial graph -> GHC + mutations -> IBIC scoring -> LCN2

    Note: Overall point is to generate a graph and improve it using different approachs
    Then test predication on the data and see which optimisation approach worked best i.e
    using the IBIC vs BIC
    """
    

    # Step 1: Load data and generate initial LCN graph 
    initial_lcn, df = generate_initial_lcn_graph(csv_data)

    # Step 2: Optimize initial graph using BIC
    bic_lcn = optimize_lcn_bic(initial_lcn, df)

    # Step 3: Optimize initial graph using IBIC
    ibic_lcn = run_lcn_ibic_optimization(initial_lcn, df)

    # print("Initial LCN: ")
    # print(initial_lcn)

    # print("BIC LCN: ")
    # print(bic_lcn)

    # print("IBIC LCN: ")
    # print(ibic_lcn)

    # Extract a Bayesian Networks from the lcn
    # BIC optimised LCN case
    bic_low_bn, bic_mid_bn, bic_high_bn = convert_lcn_to_bn(bic_lcn)

    # IBIC optimised LCN case
    ibic_low_bn, ibic_mid_bn, ibic_high_bn = convert_lcn_to_bn(ibic_lcn)


    return 