import os
import sys
import time
import torch

# Resolve project root dynamically and add src/ to python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))

from models.factory import create_model

def benchmark_model(model_name: str, device_name: str):
    device = torch.device(device_name)
    if device_name == "cuda" and not torch.cuda.is_available():
        return None
        
    config = {
        "model": {
            "name": model_name,
            "num_classes": 43,
            "pretrained": False
        }
    }
    
    try:
        model = create_model(config).to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to instantiate {model_name}: {e}")
        return None
        
    # Adjust input resolution based on model baseline settings
    img_size = 64 if model_name == "cnn_v2" else 128
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Warmup iterations to initialize PyTorch kernel operations
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
            
        if device_name == "cuda":
            torch.cuda.synchronize()
            
        start_time = time.time()
        iters = 500
        for _ in range(iters):
            _ = model(dummy_input)
            
        if device_name == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        
    avg_latency_ms = ((end_time - start_time) / iters) * 1000.0
    fps = 1000.0 / avg_latency_ms
    return avg_latency_ms, fps

def main():
    models = ["cnn_v2", "resnet50", "efficientnet_b0", "efficientnet_v2_s", "mobilenet_v3_large", "convnext_tiny"]
    devices = ["cpu", "cuda"]
    
    print("==========================================================================")
    print("  Inference Speed Benchmarking (Batch Size = 1)")
    print("==========================================================================")
    
    for device_name in devices:
        if device_name == "cuda" and not torch.cuda.is_available():
            print("\nCUDA GPU not available on this system. Skipping GPU benchmarks.\n")
            continue
            
        print(f"\nDevice: {device_name.upper()}")
        print("-" * 74)
        print(f"| {'Model Architecture':<25} | {'Resolution':<12} | {'Latency (ms)':<14} | {'FPS':<10} |")
        print("-" * 74)
        
        for model_name in models:
            res = benchmark_model(model_name, device_name)
            if res is None:
                continue
            latency, fps = res
            res_str = "64x64" if model_name == "cnn_v2" else "128x128"
            print(f"| {model_name:<25} | {res_str:<12} | {latency:>11.2f} ms | {fps:>8.1f} |")
            
        print("-" * 74)
    print("==========================================================================\n")

if __name__ == "__main__":
    main()
