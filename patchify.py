import torch
import torch.nn.functional as F
from typing import Tuple, Union, cast
from torch import Tensor

def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, int):
        return x, x
    elif isinstance(x, tuple):
        if len(x) != 2:
            raise ValueError(f"Invalid input shape, we expect a tuple of length 2. Got: {x}")
        return x
    else:
        raise ValueError(f"Invalid input type, we expect int or tuple of length 2. Got: {type(x)}")
    
def extract_tensor_patches(
    input: Tensor,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1
) -> Tensor:
    if not torch.is_tensor(input):
        raise TypeError(f"Input input type is not a Tensor. Got {type(input)}")

    if len(input.shape) != 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    window_size = cast(Tuple[int, int], _pair(window_size))
    stride = cast(Tuple[int, int], _pair(stride))

    if (stride[0] > window_size[0]) | (stride[1] > window_size[1]):
        raise AssertionError(f"Stride={stride} should be less than or equal to Window size={window_size}")

    # Pad image so that it's dimensions are divisible by the window size
    (b, c, h, w) = input.shape
    pad_h = window_size[0] - (h % window_size[0])
    pad_w = window_size[1] - (w % window_size[1])

    input_ = F.pad(input.clone(), (0, pad_w, 0, pad_h), mode="reflect")
    
    return input_

def combine_tensor_patches(
    patches: Tensor,
    original_size: Union[int, Tuple[int, int]],
    size_after_padding: Union[int, Tuple[int, int]],
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
) -> Tensor:

    if patches.ndim != 5:
        raise ValueError(f"Invalid input shape, we expect \B x N x C x H x W\. Got: {patches.shape}")

    original_size = cast(Tuple[int, int], _pair(original_size))
    size_after_padding = cast(Tuple[int, int], _pair(size_after_padding))
    window_size = cast(Tuple[int, int], _pair(window_size))
    stride = cast(Tuple[int, int], _pair(stride))

    patches = patches.flatten(2, 3).flatten(-2, -1)
    patches = patches.permute(0, 1, 3, 2)
    patches = patches.flatten(1, 2)
    
    folded = F.fold(patches, output_size=size_after_padding, kernel_size=window_size, stride=stride)
    counts = F.fold(torch.ones_like(patches), output_size=size_after_padding, kernel_size=window_size, stride=stride)


    (h, w) = original_size
    
    folded = folded / counts
    folded = folded[:, :, :h, :w]
    
    return folded