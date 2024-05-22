import json
import numpy as np
from vPull_utils import vPull_analysis

def load_json(file_path):
    """
    Load JSON data from a file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: Parsed JSON data.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    # Load properties for analysis
    properties_path = "vPull_properties.json"
    properties = load_json(properties_path)
    
    # Load sample keypoint data
    data_path = "sample_data.json"
    data = load_json(data_path)
    
    # Extract relevant information from the data
    FPS = data['FPS']
    keypoint_mapping = data['keypoint_mapping']
    pose_sequence = np.array(data['pose_sequence'])
    
    # Perform the vPull analysis
    outcome = vPull_analysis(pose_sequence, properties, FPS, keypoint_mapping)
    
    # Filter the outcome to keep only relevant metrics
    relevant_keys = [
        'pull_onset', 'pull_magnitude', '1st_step_latency', 
        '1st_step_duration', '1st_step_length', 'step_number', 
        'bending_amplitude', 'bending_latency', 'recovered', 'recover_latency'
    ]
    outcome = {key: value for key, value in outcome.items() if key in relevant_keys}
    
    # Print the analysis outcome
    print(outcome)