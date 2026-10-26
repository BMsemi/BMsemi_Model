"""
Dual-Buffer PE Architecture for Neuromorphic Accelerator
========================================================

Memory Hierarchy:
1. L1 Compute Buffer (Inside PE) - For negative weights & column differentiation
2. L2 Output Buffer (Cluster-level) - For storing output channels

Architecture:
- Each PE has:
  * Crossbar (64×64) - Stores POSITIVE weights
  * L1 Buffer (64×64) - Stores NEGATIVE weights
  * Differential computation: output = crossbar_out - L1_buffer_out

- Each Cluster has:
  * L2 Buffer - Stores accumulated output channels across layers
  * Shared among all PEs in cluster
"""

import numpy as np
from typing import Tuple, Optional, List

class L2_Buffer:
    """
    L2 Output Buffer at cluster level.
    Stores output channels from each layer.
    Shared among all PEs in a cluster.
    """
    def __init__(self, capacity: int = 1024, name: str = "L2_Buffer"):
        self.capacity = capacity
        self.name = name
        self.buffer = {}  # Dictionary: layer_id -> output_channels
        self.access_count = 0
        self.write_count = 0
        self.read_count = 0
        
    def write_output_channel(self, layer_id: int, channel_id: int, data: np.ndarray):
        """
        Write output channel to L2 buffer.
        
        Args:
            layer_id: Layer identifier
            channel_id: Channel identifier
            data: Output channel data
        """
        key = f"L{layer_id}_C{channel_id}"
        
        if len(self.buffer) >= self.capacity:
            # Eviction policy: FIFO
            oldest_key = next(iter(self.buffer))
            del self.buffer[oldest_key]
            print(f"[{self.name}] Evicted {oldest_key} (capacity full)")
        
        self.buffer[key] = data.copy()
        self.write_count += 1
        self.access_count += 1
        
    def read_output_channel(self, layer_id: int, channel_id: int) -> Optional[np.ndarray]:
        """
        Read output channel from L2 buffer.
        
        Returns:
            Output channel data or None if not found
        """
        key = f"L{layer_id}_C{channel_id}"
        
        if key in self.buffer:
            self.read_count += 1
            self.access_count += 1
            return self.buffer[key].copy()
        else:
            return None
    
    def accumulate_output(self, layer_id: int, channel_id: int, partial_data: np.ndarray):
        """
        Accumulate partial outputs into existing channel (for multi-PE accumulation).
        
        Args:
            layer_id: Layer identifier
            channel_id: Channel identifier
            partial_data: Partial output to accumulate
        """
        key = f"L{layer_id}_C{channel_id}"
        
        if key in self.buffer:
            self.buffer[key] += partial_data
        else:
            self.buffer[key] = partial_data.copy()
        
        self.write_count += 1
        self.access_count += 1
    
    def clear_layer(self, layer_id: int):
        """Clear all channels for a specific layer."""
        keys_to_remove = [k for k in self.buffer.keys() if k.startswith(f"L{layer_id}_")]
        for key in keys_to_remove:
            del self.buffer[key]
        print(f"[{self.name}] Cleared layer {layer_id}: {len(keys_to_remove)} channels")
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            'name': self.name,
            'capacity': self.capacity,
            'used': len(self.buffer),
            'access_count': self.access_count,
            'write_count': self.write_count,
            'read_count': self.read_count,
            'utilization': len(self.buffer) / self.capacity * 100
        }
    
    def clear_all(self):
        """Clear entire buffer."""
        self.buffer.clear()
        print(f"[{self.name}] All data cleared")




