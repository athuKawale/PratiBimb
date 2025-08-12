import os
import json 
from copy import deepcopy

def save_generation_data_to_json(user_id: str, generation_id: str, GENERATION_DATA: dict, file_path='data.json'):
    # Load existing data
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    # Ensure proper hierarchy
    if user_id not in data:
        data[user_id] = {}

    # Use a deep copy of generation data to avoid mutation
    temp_data = {}
    for k, v in GENERATION_DATA[generation_id].items() :
        if k != "globals" :
            temp_data[k] = v
            
    gen_data = deepcopy(temp_data)

    if generation_id not in data[user_id]:

        data[user_id][generation_id] = gen_data

        # Write to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)