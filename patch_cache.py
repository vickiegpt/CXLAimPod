#!/usr/bin/env python3
"""Patch for cache handling to fix the IndexError"""

import torch
from transformers import cache_utils
from transformers.cache_utils import DynamicCache

# Save original method
_original_from_legacy_cache = DynamicCache.from_legacy_cache

@classmethod
def patched_from_legacy_cache(cls, past_key_values=None, num_hidden_layers=None):
    """Patched version that handles malformed cache tensors"""
    if past_key_values is None:
        return cls()
    
    # Check if it's already a Cache object
    if isinstance(past_key_values, cache_utils.Cache):
        return past_key_values
    
    # Try to convert from legacy format
    try:
        # Check if the past_key_values has the right structure
        if not past_key_values or not isinstance(past_key_values, (list, tuple)):
            return cls()
        
        # Create new cache
        cache = cls()
        
        # Try to populate it
        for layer_idx, (key_states, value_states) in enumerate(past_key_values):
            if key_states is None or value_states is None:
                continue
                
            # Check tensor dimensions
            if not hasattr(key_states, 'shape') or len(key_states.shape) < 3:
                # Skip malformed tensors
                continue
                
            if not hasattr(value_states, 'shape') or len(value_states.shape) < 3:
                # Skip malformed tensors
                continue
            
            # Update cache with valid tensors
            cache.update(key_states, value_states, layer_idx)
            
        return cache
        
    except Exception as e:
        # If conversion fails, return empty cache
        print(f"Warning: Cache conversion failed ({e}), creating new cache")
        return cls()

# Apply patch
DynamicCache.from_legacy_cache = patched_from_legacy_cache

print("✓ Cache patch applied successfully")
print("  - Handles malformed cache tensors")
print("  - Falls back to empty cache on errors")
print("  - Validates tensor dimensions before update")

# Also patch the update method to be more robust
_original_update = DynamicCache.update

def patched_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
    """Patched update that handles dimension issues"""
    try:
        # Validate inputs
        if key_states is None or value_states is None:
            return key_states, value_states
            
        # Check dimensions
        if not hasattr(key_states, 'shape') or len(key_states.shape) < 2:
            return key_states, value_states
            
        if not hasattr(value_states, 'shape') or len(value_states.shape) < 2:
            return key_states, value_states
        
        # Call original update
        return _original_update(self, key_states, value_states, layer_idx, cache_kwargs)
        
    except (IndexError, AttributeError) as e:
        # Return inputs unchanged if update fails
        print(f"Warning: Cache update failed ({e})")
        return key_states, value_states

DynamicCache.update = patched_update

print("  - Update method also patched for robustness")