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
