import networkx as nx
import pandas as pd
from .sampler_helpers import generate_interval_dataset


# TODO: Check if LCNs have Type 1 and 2 Logical Constraints
# Need to have datasets for boths

def random_probability_sampler(lcn):
    """
    Samples nodes taking the random value of the interval probabilities
    """

    random_intervals_data = generate_interval_dataset(lcn)

    # Convert to DataFrame
    random_df = pd.DataFrame(random_intervals_data)

    return random_df