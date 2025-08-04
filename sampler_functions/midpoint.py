import networkx as nx
import pandas as pd
from .sampler_helpers import generate_interval_dataset


def midpoint_probability_sampler(lcn):
    """
    Samples nodes taking the midpoint value of the interval probabilities
    """

    midpoint_interval_data = generate_interval_dataset(lcn, 100, 'midpoint')

    # Convert to DataFrame
    midpoint_df = pd.DataFrame(midpoint_interval_data)

    return midpoint_df