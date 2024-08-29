import os
import glob
import pickle
import numpy as np
import re

def get_x_j_values(folder_name):
    # Extract the x and j values from the folder name "contractor_i_x_j"
    match = re.search(r'contractor_(\d+)_x_(\d+)', folder_name)
    if match:
        x_value = int(match.group(1))
        j_value = int(match.group(2))
        return x_value, j_value
    return None, None

def aggregate_tensors_from_folders(parent_directory, output_file):
    # Initialize an empty list to store thes tensors
    aggregated_tensors = []

    # Get the list of all folders in the parent_directory that match the pattern "contractor_*_*_*"
    folders_pattern = os.path.join(parent_directory, "contractor_*_x_*")
    folders = glob.glob(folders_pattern)

    # Loop through each folder and load the "embeds_attn.pkl" file
    for folder in folders:
        x_value, j_value = get_x_j_values(folder)
        if x_value is not None and j_value is not None:
            if ((x_value in [8, 9] and j_value < 50) or (x_value == 10 and j_value < 31)):
                pkl_file_path = os.path.join(folder, "embeds_attn.pkl")
        
                if os.path.exists(pkl_file_path):
                    with open(pkl_file_path, 'rb') as f:
                        tensors = pickle.load(f)
                        # Check if tensors is a list and filter out None values
                        if isinstance(tensors, list):
                            non_none_tensors = [tensor[0] for tensor in tensors if tensor is not None]
                            aggregated_tensors.extend(non_none_tensors)
                        else:
                            print(f"Warning: {pkl_file_path} does not contain a list.")
                else:
                    print(f"Warning: {pkl_file_path} does not exist.")
            else:
                print(f"Skipping folder {folder} due to x value {x_value} and j value {j_value}")

    # Convert the list of tensors to a NumPy array for easier manipulation if needed
    aggregated_tensors = np.array(aggregated_tensors)
    print(aggregated_tensors.shape)
    # Save the aggregated tensors to a single .pkl file
    with open(output_file, 'wb') as f:
        pickle.dump(aggregated_tensors, f)

    print(f"Aggregated {len(aggregated_tensors)} tensors and saved to {output_file}")

# Example usage
parent_directory = "dataset_contractor"
output_file = "aggregated_embeds_attn_clip4mc.pkl"
aggregate_tensors_from_folders(parent_directory, output_file)