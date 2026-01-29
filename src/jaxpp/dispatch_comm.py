# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# Dispatch-based communication module
# Replaces CuPy NCCL send/recv with Mori dispatch operations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

import mori
from mori.ops import EpDispatchCombineConfig, EpDispatchCombineOp
from mori.ops import EpDispatchCombineKernelType
from mori.ops import mori_shmem_init_attr

@dataclass
class DispatchTransferConfig:
    """Configuration for dispatch-based transfers between pipeline stages."""
    rank: int
    world_size: int
    hidden_dim: int  # Max size of data being transferred (flattened)
    dtype: jnp.dtype = jnp.bfloat16
    max_num_tokens_per_rank: int = 1  # For simple transfer, just 1 token
    kernel_type: str = "IntraNode"  # or "InterNode"
    gpu_per_node: int = 1


# Global handle cache - one handle per (rank, world_size, hidden_dim) config
_dispatch_handles: dict[tuple, Any] = {}
_shmem_initialized: bool = False


def _ensure_shmem_initialized(rank: int, world_size: int):
    """Initialize Mori shared memory (must be called once per process)."""
    global _shmem_initialized
    if not _shmem_initialized:
        try:
            mori_shmem_init_attr(rank, world_size)
            _shmem_initialized = True
            #print(f"Mori shmem initialized for rank {rank}/{world_size}", flush=True)
        except Exception as e:
            logger.error(f"Failed to initialize Mori shmem: {e}")
            raise


def get_dispatch_handle(config: DispatchTransferConfig) -> Any:
    """
    Get or create a dispatch handle for the given configuration.
    
    The handle is configured for "degenerate" dispatch:
    - num_experts_per_rank = 1 (one target per stage)
    - num_experts_per_token = 1 (each token goes to exactly one place)
    - This makes expert_id == target_rank for simple routing
    """
    key = (config.rank, config.world_size, config.hidden_dim, config.dtype)
    
    if key not in _dispatch_handles:
        _ensure_shmem_initialized(config.rank, config.world_size)
        
        kernel_type = getattr(
            EpDispatchCombineKernelType, 
            config.kernel_type, 
            EpDispatchCombineKernelType.IntraNode
        )
        
        print(f"{config.rank} new dispatch handle {key}", flush=True)
        mori_config = EpDispatchCombineConfig(
            data_type=config.dtype,
            rank=config.rank,
            world_size=config.world_size,
            hidden_dim=config.hidden_dim,
            scale_dim=0,  # Not needed for simple transfer
            scale_type_size=0,
            max_token_type_size=jnp.dtype(jnp.float32).itemsize,
            max_num_inp_token_per_rank=config.max_num_tokens_per_rank,
            num_experts_per_rank=1,  # Degenerate: 1 target per stage
            num_experts_per_token=1,  # Each token goes to 1 destination
            use_external_inp_buf=True,
            kernel_type=kernel_type,
            gpu_per_node=config.gpu_per_node,
        )
        
        handle = EpDispatchCombineOp(mori_config)
        _dispatch_handles[key] = handle
        logger.info(
            f"Created dispatch handle for rank {config.rank}/{config.world_size} "
            f"hidden_dim={config.hidden_dim}"
        )
    
    return _dispatch_handles[key]

def dispatch_transfer_collective(
    arrays: Sequence[jax.Array],
    my_rank: int,
    target_rank: int,
    world_size: int,
    is_sender: bool,
    shape_and_dtype: list[tuple],
) -> list[jax.Array]:

    results = []
    if is_sender and arrays:
        for arr, (shape, dtype) in zip(arrays, shape_and_dtype):
            flat_size = arr.size
            config = DispatchTransferConfig(
                rank=my_rank,
                world_size=world_size,
                hidden_dim=flat_size,
                dtype=dtype,
            )
            handle = get_dispatch_handle(config)

            # DEBUG: Print what we're sending
            # print(f"[rank {my_rank}] SEND to {target_rank}: "
            #       f"shape={arr.shape} dtype={arr.dtype} "
            #       f"first={arr.flatten()[0]} sum={float(arr.sum())}", flush=True)
            token_data = arr.reshape(1, flat_size) #.astype(dtype)
            
            # Pad to hidden_dim if needed
            # if flat_size < hidden_dim:
            #     padding = jnp.zeros((1, hidden_dim - flat_size), dtype=dtype)
            #     token_data = jnp.concatenate([token_data, padding], axis=1)
            
            # Route to target_rank
            indices = jnp.array([[target_rank]], dtype=jnp.int32)
            dummy_weights = jnp.zeros((1, 1), dtype=jnp.float32)
            dummy_scales = jnp.zeros((1, 1), dtype=jnp.float32)
            
            # TODO we shall None when scales/weights are not given
            (out_data, _, _, _, num_recv) = handle.dispatch(
                token_data, dummy_weights, dummy_scales, indices,
                has_scales=False,
                has_weights=False,
                block_num=80,
                warp_per_block=16,
            )
            
            # Sender doesn't use received data (it goes to target)
            # But we still return the original arrays for consistency
            results.append(arr)
        
        return results
    else:
        for shape, dtype in shape_and_dtype:
            flat_size = int(np.prod(shape))
            config = DispatchTransferConfig(
                rank=my_rank,
                world_size=world_size,
                hidden_dim=flat_size,
                dtype=dtype,
            )
            handle = get_dispatch_handle(config)
            
            # Receiver or observer: participate in collective
            dummy_token = jnp.zeros((1, flat_size), dtype=dtype)
            # dummy_token = jnp.full((1, flat_size), 777, dtype=dtype)
            indices = jnp.array([[target_rank]], dtype=jnp.int32)
            dummy_weights = jnp.zeros((1, 1), dtype=jnp.float32)
            dummy_scales = jnp.zeros((1, 1), dtype=jnp.float32)
        
            # Call dispatch - will receive data routed to this rank
            (out_data, _, _, _, num_recv) = handle.dispatch(
                dummy_token, dummy_weights, dummy_scales, indices,
                has_scales=False,
                has_weights=False,
                block_num=80,
                warp_per_block=16,
            )
            # IMPORTANT: Observer must return empty list (outvars=[])
            if my_rank == target_rank:
                continue
        
            arr = out_data[0, :flat_size]
            arr = arr.reshape(shape) #.astype(dtype)
            results.append(arr)
        return results

def cleanup_dispatch_handles():
    """Clean up all dispatch handles (call at program end)."""
    global _dispatch_handles, _shmem_initialized
    _dispatch_handles.clear()
    if _shmem_initialized:
        try:
            mori.shmem.shmem_finalize()
        except Exception as e:
            logger.warning(f"Error finalizing Mori shmem: {e}")
        _shmem_initialized = False
