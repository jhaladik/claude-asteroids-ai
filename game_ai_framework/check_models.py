"""
Check which models have been trained
"""

import os
import torch

def check_models():
    models_dir = "models"
    
    print("\n" + "="*60)
    print("TRAINED MODELS STATUS")
    print("="*60)
    
    if not os.path.exists(models_dir):
        print("No models directory found!")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        print("No models found!")
        return
    
    print(f"\nFound {len(model_files)} model files:\n")
    
    for model_file in sorted(model_files):
        filepath = os.path.join(models_dir, model_file)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        # Try to load and get info
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            if isinstance(checkpoint, dict):
                keys = list(checkpoint.keys())
                print(f"[OK] {model_file:<40} ({size_mb:.2f} MB)")
                print(f"     Keys: {', '.join(keys[:3])}")
                if 'task_name' in checkpoint:
                    print(f"     Task: {checkpoint['task_name']}")
            else:
                print(f"[OK] {model_file:<40} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"[ERROR] {model_file:<40} (Error loading)")
    
    print("\n" + "="*60)
    
    # Check which agents are ready
    print("\nAgent Status:")
    print("-"*30)
    
    agents = ['avoidance', 'combat', 'navigation', 'meta']
    for agent in agents:
        best_exists = os.path.exists(f"{models_dir}/best_{agent}_agent.pth")
        final_exists = os.path.exists(f"{models_dir}/{agent}_agent_final.pth")
        
        if best_exists or final_exists:
            status = "[TRAINED]"
            if best_exists and final_exists:
                status += " (best + final)"
            elif best_exists:
                status += " (best only)"
            else:
                status += " (final only)"
        else:
            status = "[NOT TRAINED]"
        
        print(f"{agent.capitalize():<15} {status}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    check_models()