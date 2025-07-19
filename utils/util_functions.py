import json

def load_json_data(filename): 
    # Open and read the JSON file
    with open(filename, 'r') as file:
        data = json.load(file)

    # return the data
    return data