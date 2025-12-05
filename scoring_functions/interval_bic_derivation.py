import numpy as np

def compute_interval_BIC(aggregate_df):
    """
    Compute interval BIC scores for each node in an aggregate contingency table.

    Matches the derivation:
        score_L(G : D) ∈ [M I_{P̲}(X;Y), M I_{P̄}(X;Y)]
        Dim[G] ∈ [Dim̲[G], Dim̄[G]]
        BIC(G : D) ∈ [M I_{P̲}(X;Y) - log(M)/2 * Dim̄[G],
                       M I_{P̄}(X;Y) - log(M)/2 * Dim̲[G]]

    Parameters
    ----------
    aggregate_df : pd.DataFrame
        Must have columns:
            ['node', 'parent_config', 'N_total',
             'count_false_lower', 'count_false_upper',
             'count_true_lower', 'count_true_upper']
    Returns
    -------
    interval_BIC : dict
        Dictionary keyed by node with values [BIC_lower, BIC_upper]
    """

    # Compute the dimensionality from contingency table
    dim_dict = compute_LCN_dimensionality(aggregate_df)

    interval_BIC = {}

    # Process each node independently
    for node in aggregate_df['node'].unique():
        df_node = aggregate_df[aggregate_df['node'] == node]

        M = df_node['N_total'].sum()  # total number of samples

        # Initialize containers for joint and marginal probabilities
        joint_lower = {}   # P̲(X,Y)
        joint_upper = {}   # P̄(X,Y)
        marginal_x_lower = {}  # P̲(X)
        marginal_x_upper = {}  # P̄(X)
        marginal_y_lower = 0   # P̲(Y=True)
        marginal_y_upper = 0   # P̄(Y=True)

        # Compute joint probabilities for each parent configuration
        for _, row in df_node.iterrows():
            parent_config = row['parent_config']
            N = row['N_total']

            # Joint probability intervals
            joint_lower[(parent_config, True)] = row['count_true_lower'] / N
            joint_upper[(parent_config, True)] = row['count_true_upper'] / N
            joint_lower[(parent_config, False)] = row['count_false_lower'] / N
            joint_upper[(parent_config, False)] = row['count_false_upper'] / N

            # Marginal of parent configuration X
            marginal_x_lower[parent_config] = N / M
            marginal_x_upper[parent_config] = N / M

            # Marginal of child Y (sum over all parent configs)
            marginal_y_lower += row['count_true_lower'] / M
            marginal_y_upper += row['count_true_upper'] / M

        # Compute interval mutual information
        score_lower = 0.0
        score_upper = 0.0
        eps = 1e-12  # avoid log(0)

        for (parent_config, y), p_xy_lower in joint_lower.items():
            p_xy_upper = joint_upper[(parent_config, y)]
            p_x_lower = marginal_x_lower[parent_config]
            p_x_upper = marginal_x_upper[parent_config]

            # CORRECTED: compute marginal of Y correctly for intervals
            if y:  # Y=True
                p_y_lower = marginal_y_lower
                p_y_upper = marginal_y_upper
            else:  # Y=False
                p_y_lower = 1 - marginal_y_upper  # lower = 1 - upper_true
                p_y_upper = 1 - marginal_y_lower  # upper = 1 - lower_true

            # Mutual information components
            score_lower += p_xy_lower * np.log((p_xy_lower + eps) / (p_x_lower * p_y_lower + eps))
            score_upper += p_xy_upper * np.log((p_xy_upper + eps) / (p_x_upper * p_y_upper + eps))

        # Multiply by M to match M * I_P(X;Y)
        MI_lower = M * score_lower
        MI_upper = M * score_upper

        # Retrieve dimensionality interval
        Dim_lower, Dim_upper = dim_dict.get(node, (0, 0))

        # Compute node-level interval BIC
        BIC_lower = MI_lower - (np.log(M) / 2) * Dim_upper  # subtract largest penalty from smallest MI
        BIC_upper = MI_upper - (np.log(M) / 2) * Dim_lower  # subtract smallest penalty from largest MI

        interval_BIC[node] = [BIC_lower, BIC_upper]

    return interval_BIC


def compute_LCN_dimensionality(aggregate_df):
    """
    Compute the dimensionality interval [Dim_lower, Dim_upper] for each node in
    an aggregate contingency table, based on the counts of child states per
    parent configuration. This produces the dim_dict required by compute_interval_BIC.

    Explanation:
    - Each parent configuration contributes (# states of node - 1) independent parameters.
      For binary nodes, each config contributes 1 parameter.
    - Upper bound (Dim̄[G]): count all parent configurations that could vary (maximal freedom).
    - Lower bound (Dim̲[G]): only count configurations where the interval of counts is non-degenerate
      (i.e., count_true_lower < count_true_upper or count_false_lower < count_false_upper).

    Parameters
    ----------
    aggregate_df : pd.DataFrame
        Must have columns:
            ['node', 'parent_config', 'N_total',
             'count_false_lower', 'count_false_upper',
             'count_true_lower', 'count_true_upper']

    Returns
    -------
    dim_dict : dict
        Dictionary keyed by node with values [Dim_lower, Dim_upper]
    """
    dim_dict = {}

    # Process each node independently
    for node in aggregate_df['node'].unique():
        df_node = aggregate_df[aggregate_df['node'] == node]

        Dim_lower = 0  # lower bound of independent parameters
        Dim_upper = 0  # upper bound of independent parameters

        for _, row in df_node.iterrows():
            # For binary nodes, one independent parameter per parent configuration
            Dim_upper += 1  # always counts toward upper bound

            # Check if the parameter is degenerate for lower bound
            # Degenerate if count_true_lower == count_true_upper or count_false_lower == count_false_upper
            # In that case, the probability is fixed, so it doesn't contribute to Dim_lower
            true_lower, true_upper = row['count_true_lower'], row['count_true_upper']
            false_lower, false_upper = row['count_false_lower'], row['count_false_upper']

            if (true_lower != true_upper) or (false_lower != false_upper):
                Dim_lower += 1  # counts toward lower bound

        dim_dict[node] = [Dim_lower, Dim_upper]

    return dim_dict


def compute_network_interval_BIC(interval_BIC):
    """
    Aggregate node-level interval BICs into a single network-level interval.

    Parameters
    ----------
    interval_BIC : dict
        Dictionary keyed by node with values [BIC_lower, BIC_upper] for each node.

    Returns
    -------
    network_interval_BIC : list of float
        [network_BIC_lower, network_BIC_upper] representing the total BIC interval
        for the entire network.
    
    Explanation
    -----------
    - BIC for a Bayesian network is additive across nodes:
        BIC_total = sum_i BIC(X_i : Parents(X_i))
    - For interval BICs:
        - The network lower bound is the sum of all node lower bounds.
        - The network upper bound is the sum of all node upper bounds.
    """
    # Sum lower bounds across all nodes
    network_BIC_lower = sum(bic[0] for bic in interval_BIC.values())

    # Sum upper bounds across all nodes
    network_BIC_upper = sum(bic[1] for bic in interval_BIC.values())

    network_interval_BIC = [network_BIC_lower, network_BIC_upper]
    return network_interval_BIC
