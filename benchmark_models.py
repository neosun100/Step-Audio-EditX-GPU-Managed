#!/usr/bin/env python3
"""
Benchmark script to compare Base vs AWQ model inference speed
"""
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark_model(model_path: str, num_iterations: int = 5):
    """Benchmark a single model"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_path}")
    print(f"{'='*60}")
    
    # Load model
    print("Loading model...")
    start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    load_time = time.time() - start_load
    print(f"✓ Model loaded in {load_time:.2f}s")
    
    # Check model dtype and device
    print(f"Model dtype: {model.dtype}")
    print(f"Model device: {model.device}")
    
    # Check if model is quantized
    if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
        print(f"Quantization config: {model.config.quantization_config}")
    
    # Warm up
    print("\nWarming up...")
    test_input = "今天天气真好，我想出去散步。"
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    
    # Benchmark inference
    print(f"\nRunning {num_iterations} iterations...")
    times = []
    for i in range(num_iterations):
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.3f}s")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 Results:")
    print(f"  Average: {avg_time:.3f}s")
    print(f"  Min:     {min_time:.3f}s")
    print(f"  Max:     {max_time:.3f}s")
    
    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return {
        'load_time': load_time,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default='/model/Step-Audio-EditX')
    parser.add_argument('--awq-model', default='/model/Step-Audio-EditX-AWQ-4bit')
    parser.add_argument('--bnb-model', default='/model/Step-Audio-EditX-bnb-4bit')
    parser.add_argument('--iterations', type=int, default=5)
    args = parser.parse_args()
    
    print("🚀 Starting Model Benchmark (Base vs AWQ vs BnB)")
    print(f"Iterations per model: {args.iterations}")
    
    # Benchmark Base model
    base_results = benchmark_model(args.base_model, args.iterations)
    
    # Benchmark AWQ model
    awq_results = benchmark_model(args.awq_model, args.iterations)
    
    # Benchmark BnB model
    bnb_results = benchmark_model(args.bnb_model, args.iterations)
    
    # Compare results
    print(f"\n{'='*80}")
    print("📈 COMPARISON (Base vs AWQ vs BnB)")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'Base':<15} {'AWQ':<15} {'BnB':<15} {'Best':<15}")
    print(f"{'-'*80}")
    
    # Load time comparison
    load_times = {'base': base_results['load_time'], 'awq': awq_results['load_time'], 'bnb': bnb_results['load_time']}
    best_load = min(load_times, key=load_times.get)
    print(f"{'Load Time':<20} {base_results['load_time']:.2f}s{'':<8} {awq_results['load_time']:.2f}s{'':<8} {bnb_results['load_time']:.2f}s{'':<8} {best_load}")
    
    # Inference time comparison
    inf_times = {'base': base_results['avg_time'], 'awq': awq_results['avg_time'], 'bnb': bnb_results['avg_time']}
    best_inf = min(inf_times, key=inf_times.get)
    print(f"{'Avg Inference':<20} {base_results['avg_time']:.3f}s{'':<8} {awq_results['avg_time']:.3f}s{'':<8} {bnb_results['avg_time']:.3f}s{'':<8} {best_inf}")
    
    # Speedup calculations
    awq_speedup = base_results['avg_time'] / awq_results['avg_time']
    bnb_speedup = base_results['avg_time'] / bnb_results['avg_time']
    
    print(f"\n{'='*80}")
    print("🏆 PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Base model:  1.00x (reference)")
    print(f"AWQ model:   {awq_speedup:.2f}x {'⚠️ SLOWER' if awq_speedup < 1 else '✅ FASTER'}")
    print(f"BnB model:   {bnb_speedup:.2f}x {'⚠️ SLOWER' if bnb_speedup < 1 else '✅ FASTER'}")
    
    if best_inf != 'base':
        print(f"\n✨ Best performer: {best_inf.upper()} model!")
    else:
        print(f"\n📝 Base model remains the fastest option.")

if __name__ == '__main__':
    main()
