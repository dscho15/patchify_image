# Project Name

## Overview

This project provides functions for extracting and combining tensor patches using PyTorch.

Features

- extract_tensor_patches: Extracts patches from a tensor with customizable window size and stride.
- combine_tensor_patches: Reconstructs the original tensor from patches, considering padding and 
size adjustments.

## Installation

To use this project, ensure you have Python 3.x installed. Clone the repository and install the requirements:

```
git clone git@github.com:dscho15/patchify_image.git
cd patchify_image
pip install -r requirements.txt
```

## Usage

Example

```
# Import the functions
from tensor_patch_operations import extract_tensor_patches, combine_tensor_patches
import torch

# Example tensor
input_tensor = torch.randn(1, 3, 64, 64)

# Extract patches
patches = extract_tensor_patches(input_tensor, window_size=(4, 4), stride=(4, 4))

# Combine patches
reconstructed_tensor = combine_tensor_patches(patches, (64, 64), (64, 64), (4, 4), (4, 4))

assert reconstructed_tensor == input_tensor
```