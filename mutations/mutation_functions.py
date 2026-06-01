"""
Functionality for performing constraint-aware mutations during structure learning

Contains functions for different contraint-aware mutations that work differently
"""


def contraint_aware_mutation():
    """
    Instead of allowing every add/remove/reverse operation, 
    mutations could first check whether the resulting structure violates 
    logical constraints.

    Examples:
    - Preventing edge removals that disconnect logically dependent nodes 
    - Prevent reversals that contradict implied logic direction
    - Enforce required edges to not break constraints

    Considerations:
    - Would need to ensure that a mutation does actually occur otherwise the search halts
    """

    return


def post_mutation_contraint_repair():
    """
    Similar to repair operators in evolutionary algorithm operators. 


    Rather than rejecting invalid mutations immediately:
    1) Perform mutation, 
    2) Detect violated logical constraints, 
    3) Repair the structure. 

    Consideration:
    Should ideally limit the number of times this can be done so it does not 
    extend the search time considerably in practice.
    """

    return


def constraint_violation_tabu_list():
    """
    Maintain a list of: 
    - Required edges
    - Prohibited edges
    - Flexible edges 

    Mutations then only operate on flexible edges

    Considerations:
    - How will the list be initialised? 
    - How will the list be updated? 
    """


    return