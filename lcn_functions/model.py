from .generator import generate_lcn
from .validator import validate_lcn
import json

"""
Generating LCNs and testing if they are valid
"""

def generate_lcn_workflow():
    # Generate small lCN (size 5)
    lcn = generate_lcn(size=5, interval_width=0.3, num_constraints=2, constraint_chaining=True)
    # print(json.dumps(lcn, indent=2))

    # Check if valid
    validation = validate_lcn(lcn)

    # TODO: Create while loop to keep running until valid

    return lcn


def create_lcn(size, interval_width):
    # Generate LCN with specified parameters
    # TODO: Use median or standard deviation for width and constraint number, num incoming edges
    lcn = generate_lcn(size, interval_width, num_constraints=2, constraint_chaining=True)
    
    return lcn
    

# Run function
generate_lcn_workflow()