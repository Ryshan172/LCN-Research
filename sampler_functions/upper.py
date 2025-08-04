import networkx as nx
import pandas as pd
from .sampler_helpers import generate_interval_dataset


def upper_probability_sampler(lcn):
    """
    Samples nodes taking the upper value of the interval probabilities
    """

    upper_interval_data = generate_interval_dataset(lcn, 100, 'lower')

    # Convert to DataFrame
    upper_df = pd.DataFrame(upper_interval_data)

    return upper_df