"""
Enhanced prefill optimization utilities for ktransformers
Includes memory-efficient chunking and gradient checkpointing
"""

import torch
import gc

def memory_efficient_prefill(model, inputs, chunk_size=4096, enable_gradient_checkpointing=False):
    """
    Memory-efficient prefill with aggressive optimization strategies:
    - Chunked processing for large sequences
    - Gradient checkpointing support
    - Explicit garbage collection
    - Cache reuse optimization
    """
    batch_size, seq_len = inputs.shape
    device = inputs.device
    
    # Ensure we're in eval mode and no grad context
    model.eval()
    
    with torch.no_grad():
        # Initialize KV cache
        past_key_values = None
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        # Enable gradient checkpointing if requested (useful for very large models)
        if enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Process in chunks
        if seq_len > chunk_size:
            # Clear any existing cache before starting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            for start_idx in range(0, seq_len, chunk_size):
                end_idx = min(start_idx + chunk_size, seq_len)
                chunk_inputs = inputs[:, start_idx:end_idx]
                chunk_position_ids = position_ids[:, start_idx:end_idx]
                
                # Get embeddings
                inputs_embeds = model.model.embed_tokens(chunk_inputs)
                
                # Forward pass with cache management
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    position_ids=chunk_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    output_attentions=False,  # Save memory
                    output_hidden_states=False  # Save memory
                )
                
                past_key_values = outputs.past_key_values
                
                # Periodic memory cleanup for very long sequences
                if (end_idx - start_idx) % (chunk_size * 4) == 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    gc.collect()
                
        else:
            # Process entire sequence at once for short sequences
            inputs_embeds = model.model.embed_tokens(inputs)
            outputs = model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False
            )
            past_key_values = outputs.past_key_values
        
        # Disable gradient checkpointing after prefill
        if enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        
        # Extract last token logits
        logits = outputs.logits
        if logits.dim() == 3:
            last_logits = logits[:, -1, :]
        else:
            last_logits = logits
            
        # Final cleanup
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return last_logits, past_key_values, seq_len


def optimize_cache_layout(past_key_values):
    """
    Optimize KV cache memory layout for better access patterns
    """
    if past_key_values is None:
        return None
    
    optimized_cache = []
    for layer_cache in past_key_values:
        if isinstance(layer_cache, tuple) and len(layer_cache) == 2:
            key_cache, value_cache = layer_cache
            # Ensure contiguous memory layout
            if key_cache is not None:
                key_cache = key_cache.contiguous()
            if value_cache is not None:
                value_cache = value_cache.contiguous()
            optimized_cache.append((key_cache, value_cache))
        else:
            optimized_cache.append(layer_cache)
    
    return tuple(optimized_cache)