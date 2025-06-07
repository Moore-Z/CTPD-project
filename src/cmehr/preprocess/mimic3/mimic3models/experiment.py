import pickle
import os
from pprint import pprint
import numpy as np

# Code for pkl
# # 1. Check file size and structure
# def quick_inspect(file_path):
#     print(f"Size: {os.path.getsize(file_path) / (1024 ** 2):.1f} MB")
#
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#         print(f"Type: {type(data)}")
#
#         if isinstance(data, dict):
#             print(f"Keys: {list(data.keys())[:5]}...")  # First 5 keys
#         elif isinstance(data, list):
#             print(f"Length: {len(data)}")
#             if data: print(f"First item: {type(data[0])}")
#         elif hasattr(data, 'shape'):
#             print(f"Shape: {data.shape}")
#
#
# # 2. Memory-efficient loading for large files
# def load_partial(file_path):  # 10MB max
#     # Load the entire pickle file
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#
#     # Check what we got
#     print(f"Type: {type(data)}")
#
#     # If it's a dictionary
#     if isinstance(data, dict):
#         print(f"Keys: {list(data.keys())}")
#         # Print some values
#         for key, value in data.items():
#             print(f"{key}: {type(value)}")
#             if hasattr(value, 'shape'):
#                 print(f"  Shape: {value.shape}")
#
#     # If it's a list
#     elif isinstance(data, list):
#         print(f"List length: {len(data)}")
#
#         # Check first item (should be a dictionary)
#         if data and isinstance(data[0], dict):
#             print("\nFirst item is a dictionary with keys:")
#             print(data[0].keys())
#
#             # Print value types for each key
#             for key, value in data[0].items():
#                 print(f"  {key}: {type(value)}")
#                 if hasattr(value, 'shape'):
#                     print(f"    Shape: {value.shape}")
#
#             # Look at specific values
#             print("\nSample values from first dictionary:")
#             for key in list(data[0].keys())[:3]:  # First 3 keys
#                 value = data[0][key]
#                 print(f"  {key}: {value}")
#
#     # If it's a numpy array or pandas DataFrame
#     elif hasattr(data, 'shape'):
#         print(f"Shape: {data.shape}")
#         print(f"Data:\n{data[:5]}")  # First 5 items
#
# # Custom print function
# def explore_data(obj, name=""):
#     if isinstance(obj, np.ndarray):
#         print(f"{name}: numpy.array {obj.shape}")
#         print(obj)
#     elif isinstance(obj, dict):
#         print(f"{name}: dict with keys: {list(obj.keys())}")
#         for k, v in obj.items():
#             explore_data(v, f"{name}.{k}")
#     elif isinstance(obj, list):
#         print(f"{name}: list[{len(obj)}]")
#         if obj: explore_data(obj[0], f"{name}[0]")
#     else:
#         print(f"{name}: {type(obj).__name__}")
#         pprint(obj)
#
# def main():
#     print("Hello, world!")
#     # Usage
#     file_path = "C:/Users/zhumo/Dataset/MIMIC3/EHR_data/mimiciii_benchmark/output_mimic3/ihm/ts_test.pkl"
#     quick_inspect(file_path)
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     explore_data(data)
#     # For large files, try partial loading
#     # data = load_partial(file_path)
#     print("Check point")

import csv
import re


# def convert_txt_to_csv(input_file_path, output_file_path):
#     """
#     Convert a text file containing CSV-like content to a proper CSV file.
#
#     Args:
#         input_file_path: Path to the input text file
#         output_file_path: Path to save the output CSV file
#     """
#     # Read the input file
#     with open(input_file_path, 'r', encoding='utf-8') as file:
#         content = file.read()
#
#     # Extract the CSV content using regex
#     # Looking for content between ```csv and ``` markers
#     csv_pattern = r'```csv\s*(.*?)\s*```'
#     matches = re.search(csv_pattern, content, re.DOTALL)
#
#     if matches:
#         csv_content = matches.group(1)
#     else:
#         # If not found between markers, try to use the whole file
#         csv_content = content
#
#     # Process the CSV content
#     lines = csv_content.strip().split('\n')
#
#     # Write to CSV file
#     with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
#         csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
#
#         for line in lines:
#             # Parse the CSV row, handling quoted fields properly
#             row = []
#             in_quotes = False
#             current_field = ''
#
#             for char in line:
#                 if char == '"':
#                     in_quotes = not in_quotes
#                 elif char == ',' and not in_quotes:
#                     row.append(current_field)
#                     current_field = ''
#                 else:
#                     current_field += char
#
#             # Add the last field
#             row.append(current_field)
#
#             # Write the row to the CSV file
#             csv_writer.writerow(row)
#
#     print(f"CSV file successfully created at {output_file_path}")



# Example usage
# if __name__ == "__main__":
#     input_file_path = "C:/Users/zhumo/Interview/LC/plan.txt"  # Path to your text file
#     output_file_path = "C:/Users/zhumo/Interview/LC/leetcode_plan.csv"  # Path to save the CSV file
#
#     convert_txt_to_csv(input_file_path, output_file_path)
import torch
if __name__ == "__main__":


    A = torch.tensor([
        [[1, 2, 3, 4]],
        [[5, 6, 7, 8]]
    ])  # shape [2, 1, 4]

    B = torch.tensor([
        [[10, 10, 10, 10],
         [20, 20, 20, 20],
         [30, 30, 30, 30]],

        [[1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3]]
    ])  # shape [2, 3, 4]

    C = A * B
    print("C.shape:", C.shape)
    print(C)
