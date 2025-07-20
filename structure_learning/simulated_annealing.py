import random
import copy
import math
from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
from pgmpy.estimators import BIC as BicScore
import networkx as nx

"""
Simulated Annealing for Structure Learning using BIC

This method searches for the best Bayesian Network structure by simulating a random walk 
through the space of possible graphs. It starts with an empty model and explores neighboring 
models by adding, removing, or reversing edges while maintaining a DAG.

At each step:
- If the new model has a better BIC score, it is accepted.
- If it is worse, it may still be accepted with a probability that decreases over time (temperature).
- This allows the algorithm to escape local maxima early in the search.

Parameters:
- initial_temp: Controls how willing the algorithm is to accept worse moves initially.
- cooling_rate: Determines how quickly the algorithm becomes more selective.
- iterations: Number of random steps to try.

Returns:
- A list of edges representing the best scoring model found.
"""

def generate_neighbors(model, nodes):
    neighbors = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
            edge = (nodes[i], nodes[j])
            reversed_edge = (edge[1], edge[0])

            # Create a copy of the current graph as a networkx.DiGraph
            G = nx.DiGraph()
            G.add_nodes_from(model.nodes())
            G.add_edges_from(model.edges())

            if model.has_edge(*edge):
                # Try removing the edge
                new_model = model.copy()
                new_model.remove_edge(*edge)
                neighbors.append(new_model)

                # Try reversing the edge
                if not model.has_edge(*reversed_edge):
                    G.remove_edge(*edge)
                    G.add_edge(*reversed_edge)
                    if nx.is_directed_acyclic_graph(G):
                        new_model = model.copy()
                        new_model.remove_edge(*edge)
                        new_model.add_edge(*reversed_edge)
                        neighbors.append(new_model)

            else:
                # Try adding the edge
                G.add_edge(*edge)
                if nx.is_directed_acyclic_graph(G):
                    new_model = model.copy()
                    new_model.add_edge(*edge)
                    neighbors.append(new_model)

    return neighbors



def creates_cycle(model, u, v):
    # Check if adding edge u -> v creates a cycle
    G = model.copy()
    G.add_edge(u, v)
    return not nx.is_directed_acyclic_graph(G)


def simulated_annealing_search(df, initial_temp=10.0, cooling_rate=0.99, iterations=1000):
    df = df.astype("category")
    nodes = list(df.columns)
    score_fn = BicScore(df)

    current_model = BayesianModel()
    current_model.add_nodes_from(nodes)
    current_score = score_fn.score(current_model)

    best_model = current_model
    best_score = current_score
    temp = initial_temp

    for i in range(iterations):
        neighbors = generate_neighbors(current_model, nodes)
        if not neighbors:
            break

        next_model = random.choice(neighbors)
        next_score = score_fn.score(next_model)

        delta = next_score - current_score
        if delta > 0 or random.random() < math.exp(delta / temp):
            current_model = next_model
            current_score = next_score
            if next_score > best_score:
                best_model = next_model
                best_score = next_score

        temp *= cooling_rate

    return best_model.edges()
