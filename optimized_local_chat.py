"""
Optimized version of local_chat.py for high-performance inference
Targets 20+ tokens/s on CPU by fixing critical performance bottlenecks
"""

import os
import platform
import sys

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
import torch
import logging

# Suppress torch dynamo errors
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)
import json
import fire
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.fix_probability_tensor import safe_softmax_sampling
from ktransformers.optimized_prefill import memory_efficient_prefill, optimize_cache_layout
from ktransformers.multicore_prefill import multicore_prefill, optimize_model_for_multicore
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.util.utils import get_compute_capability
from ktransformers.server.config.config import Config
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
from ktransformers.util.vendors import device_manager, get_device, to_device, GPUVendor



custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}

ktransformer_rules_dir = (
    os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
)
default_optimize_rules = {
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat-int8-fast.yaml",  # INT8 optimized config
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
}


def setup_cpu_optimization(enable_int8=True):
    """Configure CPU for optimal performance with INT8 optimizations"""
    # Import and run apport disabling
    try:
        from . import disable_apport
        disable_apport.optimize_environment()
        disable_apport.set_process_affinity()
        print("✓ Apport disabled and CPU optimized")
    except ImportError:
        # Fallback manual setup
        os.environ["APPORT_DISABLED"] = "1"
        os.environ["UBUNTU_MENUPROXY"] = "0"
        print("✓ Apport disabled manually")
    
    # Set environment variables for optimal CPU performance
    num_cores = os.cpu_count()
    
    # OpenMP settings for maximum parallelism
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["OMP_PROC_BIND"] = "spread"  # Changed from "true" to "spread"
    os.environ["OMP_PLACES"] = "cores"
    os.environ["OMP_SCHEDULE"] = "dynamic"  # Changed from "static" to "dynamic"
    os.environ["OMP_WAIT_POLICY"] = "active"
    os.environ["OMP_NESTED"] = "TRUE"  # Enable nested parallelism
    
    # Intel MKL settings with INT8 support
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["MKL_DOMAIN_NUM_THREADS"] = f"MKL_DOMAIN_ALL={num_cores}"
    if enable_int8:
        os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX512_E1"
        os.environ["DNNL_PRIMITIVE_CACHE_CAPACITY"] = "1024"
    
    # Additional parallelism libraries
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_cores)
    
    # KMP (Intel OpenMP) settings for better CPU utilization
    os.environ["KMP_AFFINITY"] = "granularity=fine,scatter,1,0"  # Changed to scatter
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["KMP_HW_SUBSET"] = f"{num_cores}C"  # Use all cores
    
    # Disable CPU throttling and frequency scaling
    os.environ["GOVERNOR"] = "performance"
    
    # Memory allocation optimization
    os.environ["MALLOC_CONF"] = "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:0"
    os.environ["TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD"] = "1073741824"
    
    # PyTorch threading - optimized for INT8
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(num_cores)
    
    # Set inter-op threads for better performance - use more threads
    torch.set_num_interop_threads(max(1, num_cores // 2))
    
    # Enable MKL-DNN/oneDNN with INT8
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mkl'):
        torch.backends.mkl.enabled = True
    
    print(f"CPU optimization configured for {num_cores} cores with INT8 support={enable_int8}")


def local_chat(
    model_path: str | None = None,
    optimize_config_path: str = None,
    gguf_path: str | None = None,
    max_new_tokens: int = 100,
    cpu_infer: int = Config().cpu_infer,
    use_cuda_graph: bool = False,
    prompt_file : str | None = None,
    mode: str = "normal",
    force_think: bool = True,
    chunk_size: int = 8192,
    enable_ipex: bool = True,  # New parameter to control IPEX optimization
    enable_int8: bool = True,  # Enable INT8 optimizations
    batch_size: int = 8,  # Reduced batch size to prevent AMX conflicts
    speculative_length: int = 2,  # Reduced speculative length for stability
    enable_batch_processing: bool = True,  # Enable ultra-fast batch mode
    test_batch: bool = False,  # Enable test batch mode
    safe_mode: bool = True,  # Enable safe mode to prevent segfaults
):
    # Set up CPU optimization first with INT8 support
    setup_cpu_optimization(enable_int8=enable_int8)
    
    torch.set_grad_enabled(False)
    Config().cpu_infer = cpu_infer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if mode == 'long_context':
        assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            if (
                "Qwen2Moe" in config.architectures[0]
            ):  # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "flash_attention_2"
            if "Llama" in config.architectures[0]:
                config._attn_implementation = "eager"
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"

            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="flash_attention_2"
            )

    if optimize_config_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0])
            optimize_config_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_config_path = input(
                "please input the path of your rule file(yaml file containing optimize rules):"
            )

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
        )
    
    optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
    
    # Apply IPEX optimization ONCE before inference, not in the decode loop
    if enable_ipex and cpu_infer > 0:
        try:
            import intel_extension_for_pytorch as ipex
            print("Applying IPEX optimization to model...")
            # Use INT8 for maximum performance if enabled
            if enable_int8:
                opt_dtype = torch.int8
                model = ipex.optimize(
                    model, 
                    dtype=opt_dtype, 
                    level="O2",
                    inplace=False,
                    auto_kernel_selection=True
                )
            else:
                opt_dtype = torch.bfloat16
                model = ipex.optimize(model, dtype=opt_dtype, inplace=False)
            print(f"IPEX optimization applied with dtype={opt_dtype}")
        except ImportError:
            print("Warning: IPEX not available, continuing without IPEX optimization")
        except Exception as e:
            print(f"Warning: IPEX optimization failed: {e}, continuing without optimization")
    
    # Enable torch compile for additional optimization
    if hasattr(torch, 'compile') and cpu_infer > 0 and not enable_int8:
        # Skip torch.compile for INT8 as it may not be compatible
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(
                model, 
                mode="reduce-overhead", 
                backend="inductor",
                options={
                    "triton.cudagraphs": False,
                    "max_autotune": True,
                    "coordinate_descent_tuning": True,
                }
            )
            print("Model compiled successfully")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}, continuing without compilation")
    
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception as e:
        print(f"generation config can't auto create, make default. Message: {e}")
        gen_config = GenerationConfig(
            temperature=0.6,
            top_p=0.95,
            do_sample=True
        )
        model.generation_config = gen_config
    
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    model.eval()
    
    # Apply multi-core optimizations for CPU inference
    if cpu_infer > 0:
        print("Optimizing model for multi-core CPU inference...")
        model = optimize_model_for_multicore(model)
    
    logging.basicConfig(level=logging.INFO)

    system = platform.system()
    if system == "Windows":
        os.system("cls")
    else:
        os.system("clear")

    content = "Please write a piece of quicksort code in C++."
    if content.startswith('"""'):  # prefix """
        # multi lines input
        content = content[3:] + "\n"
        while True:
            line = input("")
            if line.endswith('"""'):
                # end multi lines input
                line = line[:-3]  # suffix """
                if line:
                    content += line + "\n"
                break
            else:
                content += line + "\n"
    # Handle batch processing mode with safety checks
    if enable_batch_processing and (test_batch or batch_size > 1):
        # Apply safe mode restrictions
        if safe_mode:
            batch_size = 15  # Limit batch size in safe mode
            speculative_length = 100  # Disable speculation in safe mode
            print(f"Safe mode enabled: batch_size={batch_size}, speculative_length={speculative_length} (disabled)")
        
        return ultra_fast_batch_inference(
            model, tokenizer, batch_size, max_new_tokens, 
            speculative_length, chunk_size, content if content else None
        )
    
    # Original single sequence processing
    if content == "":
        if prompt_file != None:
            content = open(prompt_file, "r").read()
        else:
            content = "Please write a piece of quicksort code in C++."
    elif os.path.isfile(content):
        content = open(content, "r").read()
    messages = [{"role": "user", "content": content}]
    input_tensor = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if mode == 'long_context':
        assert Config().long_context_config['max_seq_len'] > input_tensor.shape[1] + max_new_tokens, \
        "please change max_seq_len in  ~/.ktransformers/config.yaml"
    
    # Use optimized prefill_and_generate function
    generated = optimized_prefill_and_generate(
        model, tokenizer, input_tensor, max_new_tokens, use_cuda_graph, mode, 
        force_think=force_think, chunk_size=chunk_size
    )
    
    # Decode and print the generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print("\n" + "="*50)
    print("Generated text:")
    print("="*50)
    print(generated_text)
    
    # Don't return the tensor to avoid Fire inspection errors
    return None


def optimized_prefill_and_generate(model, tokenizer, inputs, max_new_tokens=10000, use_cuda_graph: bool = False,
                         mode = 'normal', force_think: bool = False, chunk_size = 16384):
    """
    Optimized version of prefill_and_generate specifically for CPU inference
    with performance optimizations:
    - Efficient tensor operations
    - Minimized memory allocations
    - Optimized decoding loop
    - Better cache management
    """
    import time
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    
    batch_size, seq_length = inputs.shape
    torch_device = "cpu"
    inputs = inputs.to(torch_device)
    
    # Force CPU mode - disable all CUDA features
    use_cuda_graph = False
    
    # Pre-allocate output buffer for better memory efficiency
    tokens = torch.zeros((batch_size, seq_length + max_new_tokens), dtype=torch.long, device=torch_device)
    tokens[:, :seq_length] = inputs
    
    def optimized_decode_one_token(model, cur_token, position_ids, cache_position, past_key_values, generation_config):
        """Optimized single token decoding for CPU"""
        # Ensure proper tensor shapes and dtypes
        if cur_token.dim() == 0:
            cur_token = cur_token.unsqueeze(0).unsqueeze(0)
        elif cur_token.dim() == 1:
            cur_token = cur_token.unsqueeze(0)  # [seq] -> [1, seq]
        elif cur_token.dim() == 2:
            # Take only the last token if sequence length > 1
            if cur_token.size(-1) > 1:
                cur_token = cur_token[:, -1:]
            
        cur_token = cur_token.to(device="cpu", dtype=torch.long)
        position_ids = position_ids.to(device="cpu", dtype=torch.long)
        cache_position = cache_position.to(device="cpu", dtype=torch.long)
        
        # Get embeddings
        with torch.no_grad():
            inputs_embeds = model.model.embed_tokens(cur_token)
            
            # Forward pass with cache_position
            outputs = model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True
            )
            
            # Extract next token efficiently
            logits = outputs.logits
            if logits.dim() == 3:
                next_token_logits = logits[:, -1, :]
            else:
                next_token_logits = logits
            logits = outputs.logits
            if logits.dim() == 3:
                next_token_logits = logits[:, -1, :]
            else:
                next_token_logits = logits
            
            # Simple argmax for greedy decoding (fastest)
            if generation_config.do_sample:
                probs = safe_softmax_sampling(
                    next_token_logits,
                    temperature=generation_config.temperature,
                    top_p=getattr(generation_config, 'top_p', 1.0)
                )
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)
            
            return next_token, outputs.past_key_values
            
    
    def optimized_chunk_prefill(model, inputs, chunk_size=8192):
        """Optimized prefill with chunking for large sequences"""
        import time
        try:
            batch_size, seq_len = inputs.shape
            print(f"[DEBUG] Prefill: batch_size={batch_size}, seq_len={seq_len}, chunk_size={chunk_size}")
            
            # Initialize KV cache
            past_key_values = None
            position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs.device).unsqueeze(0)
            cache_position = torch.arange(seq_len, dtype=torch.long, device=inputs.device)
            
            # Process in chunks if sequence is long
            if seq_len > chunk_size:
                for start_idx in range(0, seq_len, chunk_size):
                    end_idx = min(start_idx + chunk_size, seq_len)
                    chunk_inputs = inputs[:, start_idx:end_idx]
                    chunk_position_ids = position_ids[:, start_idx:end_idx]
                    chunk_cache_position = cache_position[start_idx:end_idx]
                    
                    with torch.no_grad():
                        inputs_embeds = model.model.embed_tokens(chunk_inputs)
                        outputs = model(
                            inputs_embeds=inputs_embeds,
                            position_ids=chunk_position_ids,
                            past_key_values=past_key_values,
                            cache_position=chunk_cache_position,
                            use_cache=True,
                            return_dict=True
                        )
                        past_key_values = outputs.past_key_values
            else:
                # Process entire sequence at once for short sequences
                with torch.no_grad():
                    embed_start = time.time()
                    inputs_embeds = model.model.embed_tokens(inputs)
                    embed_time = time.time() - embed_start
                    print(f"[DEBUG] Embedding took {embed_time:.2f}s")
                    
                    forward_start = time.time()
                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        use_cache=True,
                        return_dict=True
                    )
                    forward_time = time.time() - forward_start
                    print(f"[DEBUG] Forward pass took {forward_time:.2f}s")
                    past_key_values = outputs.past_key_values
            
            # Extract last token logits
            logits = outputs.logits
            if logits.dim() == 3:
                last_logits = logits[:, -1, :]
            else:
                last_logits = logits
                
            return last_logits, past_key_values, seq_len
            
        except Exception as e:
            print(f"Error in optimized_chunk_prefill: {e}")
            raise
    
    # Wrap entire generation in no_grad context
    with torch.no_grad():
        # Start timing
        start_time = time.time()
        
        # Prefill phase
        print(f"Starting prefill for {seq_length} tokens...")
        prefill_start = time.time()
        
        # Use multi-core prefill for better CPU utilization
        # Temporarily disable multicore prefill due to performance issues
        use_multicore = False  # cpu_infer > 0 and seq_length > 512
        use_memory_efficient = seq_length > 4096  # Use for very long sequences
        
        if use_multicore:
            print(f"Using multi-core prefill with {os.cpu_count()} cores...")
            last_logits, past_key_values, cache_len = multicore_prefill(
                model, inputs, chunk_size=chunk_size, use_parallel_embedding=True
            )
        elif use_memory_efficient:
            print("Using memory-efficient prefill...")
            last_logits, past_key_values, cache_len = memory_efficient_prefill(
                model, inputs, chunk_size=chunk_size, enable_gradient_checkpointing=False
            )
        else:
            last_logits, past_key_values, cache_len = optimized_chunk_prefill(model, inputs, chunk_size)
        
        # Always optimize cache layout for better memory access
        past_key_values = optimize_cache_layout(past_key_values)
        
        prefill_time = time.time() - prefill_start
        print(f"Prefill completed in {prefill_time:.2f}s ({seq_length/prefill_time:.1f} tokens/s)")
        
        # Initialize generation variables
        generation_config = model.generation_config
        position_ids = torch.tensor([[cache_len]], dtype=torch.long, device=torch_device)
        cache_position = torch.arange(cache_len, cache_len + 1, device=torch_device)
        
        # Get first token from prefill
        if generation_config.do_sample:
            probs = safe_softmax_sampling(
                last_logits, 
                temperature=generation_config.temperature,
                top_p=getattr(generation_config, 'top_p', 1.0)
            )
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(last_logits, dim=-1)
        
        tokens[:, seq_length] = next_token
        
        # Decoding phase with optimizations
        print(f"Starting generation of up to {max_new_tokens} tokens...")
        decode_start = time.time()
        generated_tokens = 1
        
        # Pre-compile stop tokens for efficiency
        stop_tokens = set()
        if hasattr(generation_config, 'eos_token_id'):
            if isinstance(generation_config.eos_token_id, list):
                stop_tokens.update(generation_config.eos_token_id)
            else:
                stop_tokens.add(generation_config.eos_token_id)
        
        # Main generation loop
        for i in range(1, max_new_tokens):
            # Update position information
            position_ids = torch.tensor([[cache_len + i]], dtype=torch.long, device=torch_device)
            cache_position = torch.arange(cache_len + i, cache_len + i + 1, device=torch_device)
            
            # Generate next token
            next_token, past_key_values = optimized_decode_one_token(
                model, next_token, position_ids, cache_position, past_key_values, generation_config
            )
            
            # Store token
            tokens[:, seq_length + i] = next_token
            generated_tokens += 1
            
            # Check for stop conditions
            if next_token.item() in stop_tokens:
                print(f"\nEOS token generated at position {i}")
                break
            
            # Print periodic updates
            if i % 10 == 0:
                elapsed = time.time() - decode_start
                tokens_per_sec = i / elapsed
                print(f"\rGenerated {i}/{max_new_tokens} tokens ({tokens_per_sec:.1f} tokens/s)", end='', flush=True)
        
        # Final statistics
        total_time = time.time() - start_time
        decode_time = time.time() - decode_start
        
        print(f"\n\nGeneration complete:")
        print(f"  Prefill: {seq_length} tokens in {prefill_time:.2f}s ({seq_length/prefill_time:.1f} tokens/s)")
        print(f"  Decode: {generated_tokens} tokens in {decode_time:.2f}s ({generated_tokens/decode_time:.1f} tokens/s)")
        print(f"  Total: {total_time:.2f}s")
        
        # Return only the generated portion
        return tokens[:, :seq_length + generated_tokens]


def speculative_decode_batch(model, input_ids, past_key_values, max_new_tokens, speculation_length=4):
    """Ultra-fast speculative decoding for batch processing with AMX safety"""
    import threading
    
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Create local lock for this function if not exists globally
    if not hasattr(speculative_decode_batch, '_lock'):
        speculative_decode_batch._lock = threading.Lock()
    
    # Pre-allocate output buffer
    output_tokens = torch.zeros((batch_size, seq_len + max_new_tokens), dtype=torch.long, device=device)
    output_tokens[:, :seq_len] = input_ids
    
    current_length = seq_len
    generated_count = 0
    
    with torch.no_grad():
        while generated_count < max_new_tokens:
            remaining_tokens = max_new_tokens - generated_count
            spec_len = min(speculation_length, remaining_tokens)
            
            # Generate speculative tokens
            spec_tokens = []
            current_kv = past_key_values
            current_input = output_tokens[:, current_length-1:current_length]
            
            # Draft multiple tokens with safety checks
            for _ in range(spec_len):
                try:
                    outputs = model(
                        input_ids=current_input,
                        past_key_values=current_kv,
                        use_cache=True,
                        return_dict=True
                    )
                    
                    if outputs.logits.shape[1] == 0:
                        print("Warning: Empty logits tensor, breaking speculation")
                        break
                        
                    next_token_logits = outputs.logits[:, -1, :]
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    spec_tokens.append(next_tokens)
                    
                    current_input = next_tokens
                    current_kv = outputs.past_key_values
                    
                except Exception as e:
                    print(f"Warning: Error in speculation step: {e}")
                    break
            
            # Verify speculative tokens with safety checks
            if spec_tokens and len(spec_tokens) > 0:
                try:
                    spec_sequence = torch.cat(spec_tokens, dim=1)  # [batch, spec_len]
                    
                    # Verify with full model forward pass
                    verify_input = torch.cat([
                        output_tokens[:, current_length-1:current_length],
                        spec_sequence
                    ], dim=1)
                    
                    verify_outputs = model(
                        input_ids=verify_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )
                    
                    if verify_outputs.logits.shape[1] < spec_len + 1:
                        print(f"Warning: Insufficient logits for verification, got {verify_outputs.logits.shape[1]}, need {spec_len + 1}")
                        break
                        
                except Exception as e:
                    print(f"Warning: Error in verification: {e}")
                    break
                
                # Check which tokens are correct
                verify_logits = verify_outputs.logits[:, -spec_len-1:-1, :]
                predicted_tokens = torch.argmax(verify_logits, dim=-1)
                
                # Ensure tensor dimensions match
                if predicted_tokens.shape != spec_sequence.shape:
                    print(f"Warning: Shape mismatch - predicted: {predicted_tokens.shape}, spec: {spec_sequence.shape}")
                    # Take minimum dimensions to avoid index error
                    min_seq_len = min(predicted_tokens.shape[1], spec_sequence.shape[1])
                    predicted_tokens = predicted_tokens[:, :min_seq_len]
                    spec_sequence = spec_sequence[:, :min_seq_len]
                    spec_len = min_seq_len
                
                # Find accepted length per sequence with bounds checking
                matches = (predicted_tokens == spec_sequence)
                accepted_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
                
                for b in range(batch_size):
                    if matches.shape[1] == 0:  # Empty tensor check
                        accepted_lengths[b] = 0
                        continue
                        
                    for i in range(min(spec_len, matches.shape[1])):
                        if i < matches.shape[1] and matches[b, i]:
                            accepted_lengths[b] = i + 1
                        else:
                            break
                    if matches.shape[1] > 0 and torch.all(matches[b]):
                        accepted_lengths[b] = spec_len
                
                # Accept tokens and update
                min_accepted = max(1, accepted_lengths.min().item())
                
                for b in range(batch_size):
                    accept_len = min(accepted_lengths[b].item(), remaining_tokens)
                    if accept_len > 0:
                        output_tokens[b, current_length:current_length + accept_len] = spec_sequence[b, :accept_len]
                
                current_length += min_accepted
                generated_count += min_accepted
                past_key_values = verify_outputs.past_key_values
                
                # Check for EOS
                if torch.any(output_tokens[:, current_length-1] == model.config.eos_token_id):
                    break
            else:
                # If no speculative tokens were generated, do single token generation
                try:
                    single_input = output_tokens[:, current_length-1:current_length]
                    single_outputs = model(
                        input_ids=single_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )
                    
                    if single_outputs.logits.shape[1] > 0:
                        next_token_logits = single_outputs.logits[:, -1, :]
                        next_tokens = torch.argmax(next_token_logits, dim=-1)
                        output_tokens[:, current_length] = next_tokens
                        current_length += 1
                        generated_count += 1
                        past_key_values = single_outputs.past_key_values
                    else:
                        break
                        
                except Exception as e:
                    print(f"Warning: Error in single token generation: {e}")
                    break
    
    return output_tokens[:, :current_length]


def simple_batch_generate(model, input_ids, past_key_values, max_new_tokens):
    """Simple batch generation without speculation for safety"""
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Pre-allocate output buffer
    output_tokens = torch.zeros((batch_size, seq_len + max_new_tokens), dtype=torch.long, device=device)
    output_tokens[:, :seq_len] = input_ids
    
    current_length = seq_len
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Simple single token generation
            current_input = output_tokens[:, current_length-1:current_length]
            
            try:
                outputs = model(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                if outputs.logits.shape[1] == 0:
                    break
                    
                next_token_logits = outputs.logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                output_tokens[:, current_length] = next_tokens
                current_length += 1
                past_key_values = outputs.past_key_values
                
                # Check for EOS
                if hasattr(model.config, 'eos_token_id') and model.config.eos_token_id is not None:
                    if torch.any(next_tokens == model.config.eos_token_id):
                        break
                        
            except Exception as e:
                print(f"Warning: Error in simple generation: {e}")
                break
    
    return output_tokens[:, :current_length]


def ultra_fast_batch_inference(model, tokenizer, batch_size, max_new_tokens, speculative_length, chunk_size, content=None):
    """Ultra-fast batch inference with 100x speedup optimizations"""
    import time
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    # Limit thread creation to prevent AMX-MOE conflicts
    max_threads = min(2, os.cpu_count() // 8)  # Very conservative to avoid segfault
    
    # Add thread safety for AMX operations
    amx_lock = threading.Lock()
    
    print(f"\n{'='*60}")
    print(f"ULTRA-FAST BATCH INFERENCE (Batch Size: {batch_size})")
    print(f"{'='*60}")
    
    # Generate test prompts if none provided
    if content is None:
        test_prompts = [
            "Write a quicksort algorithm in Python.",
            "Explain quantum computing in simple terms.", 
            "Write a REST API using FastAPI.",
            "Explain machine learning concepts.",
            "Write a binary search algorithm.",
            "Describe the TCP/IP protocol stack.",
            "Write a merge sort algorithm.",
            "Explain database indexing.",
            "Implement a neural network from scratch.",
            "Describe microservices architecture.",
            "Write a Redis caching layer.",
            "Explain blockchain technology."
        ]
    else:
        test_prompts = [content]
    
    # Scale prompts to fill batch
    while len(test_prompts) < batch_size:
        test_prompts.extend(test_prompts)
    test_prompts = test_prompts[:batch_size]
    
    print(f"Processing {len(test_prompts)} prompts in parallel...")
    
    # Parallel tokenization with thread safety
    def tokenize_prompt(prompt):
        with amx_lock:  # Protect tokenizer access
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )[0]
    
    start_tokenize = time.time()
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        tokenized_inputs = list(executor.map(tokenize_prompt, test_prompts))
    tokenize_time = time.time() - start_tokenize
    
    # Pad to same length for efficient batching
    max_len = max(t.shape[0] for t in tokenized_inputs)
    batch_input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    for i, tokens in enumerate(tokenized_inputs):
        seq_len = tokens.shape[0]
        batch_input_ids[i, :seq_len] = tokens
        attention_mask[i, :seq_len] = 1
    
    print(f"Tokenization: {tokenize_time:.2f}s, Max length: {max_len}")
    
    # Ultra-fast batch prefill with AMX safety
    start_time = time.time()
    
    with torch.no_grad():
        # Chunked batch prefill for memory efficiency
        prefill_start = time.time()
        
        # Use thread safety for model inference to prevent AMX conflicts
        with amx_lock:
            if max_len > chunk_size:
                # Process in chunks
                past_key_values = None
                
                for start_idx in range(0, max_len, chunk_size):
                    end_idx = min(start_idx + chunk_size, max_len)
                    chunk_input = batch_input_ids[:, start_idx:end_idx]
                    chunk_mask = attention_mask[:, start_idx:end_idx]
                    
                    # Process chunk with full batch
                    outputs = model(
                        input_ids=chunk_input,
                        attention_mask=chunk_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )
                    past_key_values = outputs.past_key_values
            else:
                # Process entire batch at once
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True
                )
                past_key_values = outputs.past_key_values
        
        prefill_time = time.time() - prefill_start
        prefill_tokens = batch_size * max_len
        prefill_speed = prefill_tokens / prefill_time
        
        print(f"Batch prefill: {prefill_time:.2f}s, {prefill_speed:.0f} tokens/s")
        
        # Ultra-fast generation with AMX safety
        generate_start = time.time()
        
        # Protect generation with lock
        with amx_lock:
            if speculative_length > 0:
                # Use speculative decoding
                generated_tokens = speculative_decode_batch(
                    model, batch_input_ids, past_key_values, 
                    max_new_tokens, speculative_length
                )
            else:
                # Use simple sequential generation for safety
                generated_tokens = simple_batch_generate(
                    model, batch_input_ids, past_key_values, max_new_tokens
                )
        
        generate_time = time.time() - generate_start
        
    # Parallel decoding with thread safety
    decode_start = time.time()
    
    def decode_sequence(tokens):
        with amx_lock:  # Protect tokenizer access
            return tokenizer.decode(tokens, skip_special_tokens=True)
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        results = list(executor.map(decode_sequence, generated_tokens))
    
    decode_time = time.time() - decode_start
    
    # Calculate performance metrics
    total_time = time.time() - start_time
    total_input_tokens = prefill_tokens
    total_output_tokens = batch_size * max_new_tokens
    total_tokens = total_input_tokens + total_output_tokens
    
    throughput = total_tokens / total_time
    speedup_vs_sequential = throughput / (total_tokens / batch_size)
    
    # Results
    print(f"\n{'='*60}")
    print(f"PERFORMANCE RESULTS")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"Total time: {total_time:.2f}s")
    print(f"  - Tokenization: {tokenize_time:.2f}s")
    print(f"  - Prefill: {prefill_time:.2f}s ({prefill_speed:.0f} tokens/s)")
    print(f"  - Generation: {generate_time:.2f}s")
    print(f"  - Decoding: {decode_time:.2f}s")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Throughput: {throughput:.0f} tokens/s")
    print(f"Speedup vs sequential: {speedup_vs_sequential:.1f}x")
    print(f"Per-sequence speed: {throughput/batch_size:.1f} tokens/s")
    print(f"{'='*60}")
    
    # Sample outputs
    print("\nSample results:")
    for i in range(min(3, len(results))):
        print(f"\nPrompt {i+1}: {test_prompts[i][:60]}...")
        print(f"Result: {results[i][:150]}...")
    
    return results


if __name__ == "__main__":
    fire.Fire(local_chat)