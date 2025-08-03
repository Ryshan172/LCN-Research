# LCN and Dataset Overview 

## Created LCNs for testing 

| Name               | Size   | Nodes | Logical Constraints |
|--------------------|--------|-------|----------------------|
| Smoking            | Small  | 3     | 1                    |
| Smoking            | Medium | 7     | 4                    |
| Psychiatry         | Small  | 5     | 1                    |
| Psychiatry         | Medium | 8     | 3                    |
| Climate            | Small  | 4     | 1                    |
| Climate            | Medium | 10    | 6                    |


## Data Sampling
To create a datasets for Structure Learning, Heuristic Optimisation and Probabilistic Circuits, the interval values of the nodes of each LCN will be sampled in the following ways:

- Lower Probability: Sampling nodes taking the lower value of the interval probabilities
- Upper Probability: Sampling nodes taking the upper value of the interval probabilities
- Random: Sampling nodes taking the random value of the interval probabilities
- Midpoint: Sampling nodes taking the midpoint value of the interval probabilities