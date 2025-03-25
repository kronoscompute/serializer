import unittest
import torch.nn as nn
import json
import pickle
import torch
import csv
import os
from serializer import serialize_model_and_datafunction  # Replace 'your_module' with the actual module name

# Example PyTorch model for testing
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Example data generator function for testing
def sample_datagenerator():
    return torch.randn(1, 784)

class TestSerializeModelAndDataFunction(unittest.TestCase):
    def setUp(self):
        # Setup test data
        self.model = SimpleNet()
        self.datagenerator = sample_datagenerator
        self.string_list = ["apple", "banana", "cherry"]
        self.result = serialize_model_and_datafunction(self.model, self.datagenerator, self.string_list)

    def tearDown(self):
        # Clean up files after each test
        for filename in self.result.values():
            if os.path.exists(filename):
                os.remove(filename)

    def test_files_created(self):
        # Test that all three files are created
        self.assertTrue(os.path.exists(self.result["model_file"]))
        self.assertTrue(os.path.exists(self.result["datagenerator_file"]))
        self.assertTrue(os.path.exists(self.result["data_file"]))

    def test_model_architecture_json(self):
        # Test the JSON file contains the correct architecture
        with open(self.result["model_file"], "r") as f:
            architecture = json.load(f)
        
        layers = architecture["layers"]
        self.assertEqual(len(layers), 3)  # fc1, relu, fc2
        
        # Test fc1 layer
        self.assertEqual(layers[0]["name"], "fc1")
        self.assertEqual(layers[0]["type"], "Linear")
        self.assertEqual(layers[0]["in_features"], 784)
        self.assertEqual(layers[0]["out_features"], 256)
        
        # Test relu layer
        self.assertEqual(layers[1]["name"], "relu")
        self.assertEqual(layers[1]["type"], "ReLU")
        
        # Test fc2 layer
        self.assertEqual(layers[2]["name"], "fc2")
        self.assertEqual(layers[2]["type"], "Linear")
        self.assertEqual(layers[2]["in_features"], 256)
        self.assertEqual(layers[2]["out_features"], 10)

    def test_datagenerator_pickle(self):
        # Test the pickle file contains the correct function
        with open(self.result["datagenerator_file"], "rb") as f:
            loaded_func = pickle.load(f)
        
        # Check that the loaded function is callable and returns a tensor
        self.assertTrue(callable(loaded_func))
        result = loaded_func()
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 784))

    def test_strings_csv(self):
        # Test the CSV file contains the correct string list
        with open(self.result["data_file"], "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        self.assertEqual(len(rows), 1)  # Single row
        self.assertEqual(rows[0], self.string_list)

    def test_filename_uniqueness(self):
        # Test that running the function twice generates different filenames
        result2 = serialize_model_and_datafunction(self.model, self.datagenerator, self.string_list)
        
        # Clean up second set of files
        for filename in result2.values():
            if os.path.exists(filename):
                os.remove(filename)
        
        self.assertNotEqual(self.result["model_file"], result2["model_file"])
        self.assertNotEqual(self.result["datagenerator_file"], result2["datagenerator_file"])
        self.assertNotEqual(self.result["data_file"], result2["data_file"])