import torch.nn as nn
import json
import pickle
import csv
import uuid

def serialize_model_and_datafunction(model, datagenerator, data_urls):
    """
    Serializes a PyTorch model's architecture to JSON, a datagenerator function to pickle,
    and a list of strings to a CSV file, using UUID-based filenames.
    
    Args:
        model: PyTorch model (nn.Module subclass)
        datagenerator: Python function to generate data
        data_urls: List of strings to save in a CSV file
    
    Returns:
        dict: Dictionary containing the filenames of the serialized files
    """
    # Generate unique UUIDs for filenames
    model_uuid = str(uuid.uuid4())
    datagen_uuid = str(uuid.uuid4())
    csv_uuid = str(uuid.uuid4())

    # 1. Serialize model architecture to JSON
    architecture = {"layers": []}
    for name, module in model.named_modules():
        if name == "":  # Skip the top-level module itself
            continue
        layer_info = {"name": name, "type": module.__class__.__name__}
        
        # Add specific parameters based on layer type
        if isinstance(module, nn.Linear):
            layer_info["in_features"] = module.in_features
            layer_info["out_features"] = module.out_features
        elif isinstance(module, nn.Conv2d):
            layer_info["in_channels"] = module.in_channels
            layer_info["out_channels"] = module.out_channels
            layer_info["kernel_size"] = module.kernel_size
        # Add more layer types as needed
        
        architecture["layers"].append(layer_info)
    
    model_filename = f"model_architecture_{model_uuid}.json"
    with open(model_filename, "w") as f:
        json.dump(architecture, f, indent=4)

    # 2. Serialize datagenerator function to pickle
    datagen_filename = f"datagenerator_{datagen_uuid}.pkl"
    with open(datagen_filename, "wb") as f:
        pickle.dump(datagenerator, f)

    # 3. Serialize data_urls to CSV (single row)
    csv_filename = f"data_{csv_uuid}.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data_urls)

    # Return filenames for reference
    return {
        "model_file": model_filename,
        "datagenerator_file": datagen_filename,
        "data_file": csv_filename
    }