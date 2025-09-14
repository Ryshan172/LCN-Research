from .generator import generate_lcn
from .validator import validate_lcn
import json
from .lcn_check import validate_generated_lcn
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
    
    # Running in loop to ensure only valid LCN is outputted
    
    is_valid = False
    attempts = 0
    max_attempts=10

    while not is_valid and attempts < max_attempts:
        attempts += 1

        # Generate candidate LCN
        lcn = generate_lcn(size, interval_width, num_constraints=2, constraint_chaining=True)

        # Validate candidate LCN
        is_valid = validate_generated_lcn(lcn)

        print(f"Attempt {attempts}: Is Valid? {is_valid}")

    if not is_valid:
        raise RuntimeError(f"Failed to generate valid LCN after {max_attempts} attempts.")

    return lcn
    

# Run function
generate_lcn_workflow()