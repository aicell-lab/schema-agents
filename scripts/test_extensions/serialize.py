import json
from typing import Any
import pickle as pkl
import sys

def convert_to_serializable(data):
    if isinstance(data, bytes):
        return data.decode('utf-8')
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif hasattr(data, 'dict'):  # Check if it's a Pydantic model instance
        return convert_to_serializable(data.dict())
    return data

def dump_metadata_json(metadata, file_path: str):
    serialized_metadata = convert_to_serializable(metadata)
    with open(file_path, 'w') as f:
        json.dump(serialized_metadata, f)

# if __name__ == "__main__":
#     metadata = pkl.load(open("/Users/gkreder/schema-agents/test.pkl", "rb"))
#     # Assuming `steps` is your complex nested list that you've mentioned
#     steps_serializable = convert_to_serializable(metadata['steps'])
#     # Now you can safely dump this to a JSON file
#     with open('steps.json', 'w') as f:
#         json.dump(steps_serializable, f)
#     print(metadata)
