import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["JOBLIB_START_METHOD"] = "spawn"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random
import math
import networkx as nx
from simanneal import Annealer
from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
from pgmpy.estimators import BIC


"""
Using the Simanneal packacge to run simulated annealing on the 
forward sample bayesian network dataset
"""

# Prevent joblib from using multiprocessing (the cause of the signal error)
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["JOBLIB_START_METHOD"] = "threading"


class BNAnnealer(Annealer):
    """
    A Simanneal-compatible class that defines how to:
    - represent the "state" (a BayesianModel)
    - compute the "energy" (negative BIC, since SA minimizes energy)
    - generate "moves" (local structure modifications)

    SimAnneal handles:
    - the cooling schedule
    - acceptance probability
    - optimization loop
    """

    def __init__(self, initial_model, df):
        """
        Parameters
        ----------
        initial_model : BayesianModel
            The starting point of the simulated annealing search.
            Typically an empty or random DAG.

        df : pandas.DataFrame
            The dataset forward-sampled from the true Bayesian Network.
            Used to compute the BIC score of each candidate structure.
        """

        # Fix for pgmpy scoring inside simanneal threads
        os.environ["JOBLIB_MULTIPROCESSING"] = "0"
        os.environ["JOBLIB_START_METHOD"] = "threading"

        # Store dataset
        self.df = df

        # List of nodes (BN variables)
        self.nodes = list(df.columns)

        # Initialize the BIC scoring function
        self.score_fn = BIC(df)

        # This sets the starting state for the annealing algorithm
        super().__init__(initial_model)


    # ENERGY FUNCTION
    def energy(self):
        """
        Returns the "energy" of the current state.

        SimAnneal *minimizes* energy.
        Since BIC is normally *maximized*, we return negative BIC.

        If the model is invalid (e.g. contains illegal CPDs), we assign a
        very high penalty so the algorithm rejects that state.
        """

        try:
            # Higher BIC is better → lower energy
            return -self.score_fn.score(self.state)
        except:
            # If model cannot be scored, assign huge penalty
            return 1e9


    # MOVE OPERATOR
    def move(self):
        """
        Proposes a random neighboring structure by:
        - adding an edge,
        - removing an edge, or
        - reversing an existing edge.

        The new structure must remain a DAG (acyclic).
        """

        model = self.state

        # Pick two distinct nodes for a potential edge modification
        u, v = random.sample(self.nodes, 2)

        if model.has_edge(u, v):
            # --- Case 1: Edge exists (u → v)
            # 50% chance: remove edge
            if random.random() < 0.5:
                model.remove_edge(u, v)

            else:
                # Try reversing the edge: u→v becomes v→u
                model.remove_edge(u, v)

                # Only reverse if the opposite edge does not already exist
                if not model.has_edge(v, u):
                    model.add_edge(v, u)

                    # Must remain acyclic → otherwise undo
                    if not nx.is_directed_acyclic_graph(model):
                        model.remove_edge(v, u)
                        model.add_edge(u, v)

        else:
            # --- Case 2: Edge does NOT exist
            # Try adding u → v
            model.add_edge(u, v)

            # Revert addition if it introduces a cycle
            if not nx.is_directed_acyclic_graph(model):
                model.remove_edge(u, v)

        # Update the internal state
        self.state = model


# RUNNING THE OPTIMIZER
def run_simanneal_sa(df, steps=20000):
    """
    Perform simulated annealing to learn a Bayesian Network structure.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset used for scoring candidate structures.

    steps : int
        Number of temperature steps in the annealing schedule.
        More steps = better chance of reaching global optimum (but slower).

    Returns
    -------
    best_model : BayesianModel
        The best structure found during the annealing process.

    best_score : float
        The corresponding BIC score of that structure.
    """

    # Start with an empty model with all nodes but no edges
    initial_model = BayesianModel()
    initial_model.add_nodes_from(df.columns)

    # Create our annealer using the BNAnnealer class
    annealer = BNAnnealer(initial_model, df)

    # Optional tuning: overall number of steps and temperature range
    annealer.steps = steps        # number of iterations
    annealer.Tmax = 1.0           # starting temperature
    annealer.Tmin = 0.001         # ending temperature

    # Run the annealing process
    best_model, best_energy = annealer.anneal()

    # Convert energy back into BIC score
    best_score = -best_energy

    sa_edges = best_model.edges()
    sa_score = best_score

    return sa_edges, sa_score