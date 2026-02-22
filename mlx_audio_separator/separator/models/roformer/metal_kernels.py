"""
Custom Metal kernels for HTDemucs MLX optimization.

These kernels fuse operations to reduce memory bandwidth and improve performance
while maintaining numerical parity with PyTorch.
"""

import mlx.core as mx

# Simple elementwise normalize kernel
# Applies pre-computed mean/std in a single efficient pass
FUSED_NORMALIZE_SOURCE = '''
    uint idx = thread_position_in_grid.x;

    uint total_elements = input_shape[0];

    if (idx >= total_elements) return;

    float mean_val = mean_array[0];
    float std_val = std_array[0];
    float eps_val = epsilon_array[0];

    output[idx] = (input[idx] - mean_val) / (std_val + eps_val);
'''


# Cache compiled kernels
_kernel_cache = {}


def _get_or_compile_normalize_kernel():
    """Get or compile the fused normalization kernel."""
    if 'fused_normalize' not in _kernel_cache:
        _kernel_cache['fused_normalize'] = mx.fast.metal_kernel(
            name="fused_normalize",
            input_names=["input", "mean_array", "std_array", "epsilon_array"],
            output_names=["output"],
            source=FUSED_NORMALIZE_SOURCE,
            ensure_row_contiguous=True,
        )
    return _kernel_cache['fused_normalize']


def fused_normalize(x: mx.array, mean: mx.array, std: mx.array, eps: float = 1e-5) -> mx.array:
    """
    Apply normalization with pre-computed mean/std using custom Metal kernel.

    Computes: (x - mean) / (std + eps)

    This fuses the three operations (subtract, add, divide) into a single Metal kernel,
    reducing memory bandwidth.

    Args:
        x: Input tensor (any shape, will be flattened)
        mean: Pre-computed mean (scalar array, shape (1,))
        std: Pre-computed std (scalar array, shape (1,))
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor with same shape as input

    Performance:
        - Standard MLX: 3 separate operations with 3 memory passes
        - This kernel: Single fused operation with 1 memory pass
        - Expected speedup: ~1.2-1.5x on large tensors

    Parity:
        - Identical algorithm, just fused
        - Expected max_abs error: 0 (bit-exact)
    """
    original_shape = x.shape
    x_flat = mx.reshape(x, (-1,))
    total_elements = x_flat.size

    kernel = _get_or_compile_normalize_kernel()

    # Ensure scalars are arrays with shape (1,)
    mean_array = mx.array([float(mean)], dtype=mx.float32) if mean.size == 1 else mean
    std_array = mx.array([float(std)], dtype=mx.float32) if std.size == 1 else std
    eps_array = mx.array([eps], dtype=mx.float32)

    grid = (total_elements, 1, 1)
    threadgroup = (256, 1, 1)

    outputs = kernel(
        inputs=[x_flat, mean_array, std_array, eps_array],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(total_elements,)],
        output_dtypes=[x.dtype],
    )

    return mx.reshape(outputs[0], original_shape)
