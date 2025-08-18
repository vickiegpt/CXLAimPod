"""
Multi-core optimized prefill for CPU inference
Ensures all CPU cores are utilized during prefill phase
"""

import torch
import os
import gc
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def setup_multicore_environment():
    """Enhanced CPU optimization for multi-core prefill"""
    num_cores = os.cpu_count()
    
    # Aggressive multi-threading settings
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_cores)
    
    # Intel MKL specific optimizations
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["MKL_VERBOSE"] = "1"
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_SETTINGS"] = "1"
    
    # PyTorch threading
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(num_cores)
    
    # Enable MKL-DNN (now oneDNN)
    if hasattr(torch, '_C') and hasattr(torch._C, '_set_mkldnn_enabled'):
        torch._C._set_mkldnn_enabled(True)
    
    print(f"Multi-core prefill configured for {num_cores} cores")
    return num_cores


def parallel_embed_tokens(model, input_chunks, num_workers=None):
    """Parallelize token embedding across CPU cores"""
    if num_workers is None:
        num_workers = os.cpu_count()
    
    def embed_chunk(chunk):
        with torch.no_grad():
            return model.model.embed_tokens(chunk)
    
    # For small sequences, parallel processing overhead isn't worth it
    if len(input_chunks) == 1:
        return embed_chunk(input_chunks[0])
    
    # Use ThreadPoolExecutor for parallel embedding
    with ThreadPoolExecutor(max_workers=min(num_workers, len(input_chunks))) as executor:
        futures = [executor.submit(embed_chunk, chunk) for chunk in input_chunks]
        embeddings = [future.result() for future in futures]
    
    return torch.cat(embeddings, dim=1)


def multicore_attention_forward(model, inputs_embeds, position_ids, past_key_values=None, 
                               chunk_size=512, num_workers=None, use_cache=True):
    """
    Multi-core optimized attention forward pass
    Splits computation across cores for better parallelism
    """
    if num_workers is None:
        num_workers = os.cpu_count()
    
    batch_size, seq_len, hidden_dim = inputs_embeds.shape
    
    # For very long sequences, use parallel chunk processing
    if seq_len > chunk_size * 2:
        chunks = []
        chunk_positions = []
        
        for i in range(0, seq_len, chunk_size):
            end = min(i + chunk_size, seq_len)
            chunks.append(inputs_embeds[:, i:end, :])
            chunk_positions.append(position_ids[:, i:end])
        
        # Process chunks in parallel batches
        batch_size_per_worker = max(1, len(chunks) // num_workers)
        
        def process_chunk_batch(chunk_batch, pos_batch):
            results = []
            for chunk, pos in zip(chunk_batch, pos_batch):
                output = model(
                    inputs_embeds=chunk,
                    position_ids=pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False
                )
                results.append(output)
            return results
        
        # Split chunks across workers
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(0, len(chunks), batch_size_per_worker):
                end = min(i + batch_size_per_worker, len(chunks))
                futures.append(
                    executor.submit(
                        process_chunk_batch,
                        chunks[i:end],
                        chunk_positions[i:end]
                    )
                )
            
            # Collect results
            all_outputs = []
            for future in futures:
                all_outputs.extend(future.result())
        
        # Combine outputs
        logits = torch.cat([out.logits for out in all_outputs], dim=1)
        past_key_values = all_outputs[-1].past_key_values
        
        return logits, past_key_values
    
    else:
        # For shorter sequences, use standard forward pass
        outputs = model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False
        )
        return outputs.logits, outputs.past_key_values


def multicore_prefill(model, inputs, chunk_size=4096, use_parallel_embedding=True):
    """
    Enhanced prefill that utilizes all CPU cores
    """
    # Setup multi-core environment
    num_cores = setup_multicore_environment()
    
    batch_size, seq_len = inputs.shape
    device = inputs.device
    
    model.eval()
    
    with torch.no_grad():
        # Enable MKLDNN for CPU inference
        if device.type == 'cpu' and hasattr(torch, 'backends'):
            torch.backends.mkldnn.enabled = True
            torch.backends.mkldnn.verbose = True
        
        # Initialize position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        # Parallel token embedding if sequence is long
        # Disable parallel embedding for now as it may cause issues
        if False and use_parallel_embedding and seq_len > 1024:
            print(f"Using parallel embedding with {num_cores} workers...")
            # Split input for parallel embedding
            chunk_size_embed = seq_len // num_cores
            input_chunks = []
            for i in range(0, seq_len, chunk_size_embed):
                end = min(i + chunk_size_embed, seq_len)
                input_chunks.append(inputs[:, i:end])
            
            inputs_embeds = parallel_embed_tokens(model, input_chunks, num_cores)
        else:
            print(f"Using standard embedding (sequence length: {seq_len})...")
            inputs_embeds = model.model.embed_tokens(inputs)
        
        # For now, use standard forward pass with optimized settings
        # Multi-core chunk processing can cause issues with some model architectures
        print(f"Running optimized forward pass with {num_cores} threads...")
        outputs = model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False
        )
        
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        
        # Extract last token logits
        if logits.dim() == 3:
            last_logits = logits[:, -1, :]
        else:
            last_logits = logits
        
        # Force synchronization
        if device.type == 'cpu':
            torch.cpu.synchronize()
        
        return last_logits, past_key_values, seq_len


def optimize_model_for_multicore(model):
    """
    Apply model-level optimizations for multi-core CPU inference
    """
    # Enable channels_last memory format for better cache efficiency
    if hasattr(model, 'to_channels_last'):
        model = model.to(memory_format=torch.channels_last)
    
    # Enable graph optimization if available
    if hasattr(torch, 'jit') and hasattr(torch.jit, 'optimize_for_inference'):
        try:
            model = torch.jit.optimize_for_inference(model)
        except:
            pass
    
    # Set model to eval mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    return model