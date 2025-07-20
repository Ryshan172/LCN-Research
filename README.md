# Setup
- Activate virtual environment 
- pip install -r requirements.txt

## Virtual environment
python -m venv code_env

source code_env/bin/activate


## LCN Structures
- Using JSON e.g. net1.json
- programmatic representation for:
    - DAG structure
    - Conditional Credal Sets (interval-valued CPTs)
    - Logical constraints (e.g., if A=True then B=False)
- JSON schema for each network instance 

JSON supports:
- DAG structure
- Conditional credal sets
- Logical constraints


# LCN Datasets

## Sampling From The Network
Because probabilities are intervals, trying different sampling strategies:
- Midpoint of intervals
- Random value within interval
- Lower bound or upper bound

Each sample is generated topologically:
- Sample root node using interval strategy
- Recursively sample child nodes conditioned on parent value(s)

### Implementation Ideas
Topological sort of the DAG

For each node:
- Select probability from interval using a strategy
- Sample a value using that probability

Check if any logical constraint is violated → reject and resample if necessary (rejection sampling)

Store the final dataset as a CSV with each row being one sample.\


# Useful Links
https://github.com/IBM/LCN



# LCN Structures

## Net1

This Logical Credal Network (LCN) models three binary variables: A, B, and C. The relationships are as follows:

A is a root node with uncertainty represented by an interval-valued prior:
- P(A=True) ∈ [0.4, 0.6]
- P(A=False) ∈ [0.4, 0.6]

B is conditionally dependent on A:

If A=True:

- P(B=True) ∈ [0.3, 0.7]
- P(B=False) ∈ [0.3, 0.7]

If A=False:

- P(B=True) ∈ [0.6, 0.9]
- P(B=False) ∈ [0.1, 0.4]

C is conditionally dependent on B:

If B=True:

- P(C=True) ∈ [0.7, 0.9]
- P(C=False) ∈ [0.1, 0.3]

If B=False:

- P(C=True) ∈ [0.2, 0.4]
- P(C=False) ∈ [0.6, 0.8]

In addition to probabilistic uncertainty, the network encodes a logical constraint:

If A is true, then B must be false.

This logical rule overrides or filters out probabilistic configurations that would otherwise violate the implication, effectively restricting the set of admissible joint distributions.