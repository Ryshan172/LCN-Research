import pandas as pd

# Helper to regroup contingency rows
def regroup_for_parents(full_table, node, parent_set):
    """
    Regroup the aggregate contingency table for the given node and candidate parent_set.
    This produces exactly one row per unique parent configuration in `parent_set`.
    """
    df_node = full_table[full_table['node'] == node].copy()

    # Parse parent_config strings into dict
    def parse_config(config_str):
        if config_str == "[]" or config_str.strip() == "":
            return {}
        d = {}
        for pair in config_str.split(","):
            key, val = pair.split("=")
            key = key.strip()
            val = val.strip()
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            d[key] = val
        return d

    df_node['parent_config_dict'] = df_node['parent_config'].apply(parse_config)

    # Group rows according to candidate parent_set
    grouped = {}
    for _, row in df_node.iterrows():
        config = row['parent_config_dict']
        # Extract only the parents in the candidate set
        key = tuple(sorted((p, config.get(p, False)) for p in parent_set))
        if key not in grouped:
            grouped[key] = {
                'N_total': 0,
                'count_true_lower': 0,
                'count_true_upper': 0,
                'count_false_lower': 0,
                'count_false_upper': 0
            }
        grouped[key]['N_total'] += row['N_total']
        grouped[key]['count_true_lower'] += row['count_true_lower']
        grouped[key]['count_true_upper'] += row['count_true_upper']
        grouped[key]['count_false_lower'] += row['count_false_lower']
        grouped[key]['count_false_upper'] += row['count_false_upper']

    # Convert to list of rows compatible with compute_interval_BIC
    regrouped_rows = []
    for key, counts in grouped.items():
        # convert key tuple back to string for compatibility
        if len(key) == 0:
            key_str = "[]"
        else:
            key_str = ", ".join(f"{p}={v}" for p, v in key)
        row_dict = {
            'node': node,
            'parent_config': key_str,
            **counts
        }
        regrouped_rows.append(row_dict)

    return pd.DataFrame(regrouped_rows)