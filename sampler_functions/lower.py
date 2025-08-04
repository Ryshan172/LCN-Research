import networkx as nx
import pandas as pd
from .sampler_helpers import generate_interval_dataset


def lower_probability_sampler(lcn):
    """
    Samples nodes taking the lower value of the interval probabilities
    """

    lower_interval_data = generate_interval_dataset(lcn, 100, 'lower')

    # Convert to dataframe
    lower_interval_df = pd.DataFrame(lower_interval_data)

    return lower_interval_df