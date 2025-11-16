#!/usr/bin/env python3
"""
Script to add noise to model weights to escape local minimum
"""

import torch
import numpy as np

def add_noise_to_checkpoint():
    # Load checkpoint
    checkpoint = torch.load("checkpoints/latest_checkpoint.pth")
    
    print("Adding noise to model weights to encourage exploration...")
    
    # Add small random noise to all weights
    state_dict = checkpoint['model_state_dict']
    
    for name, param in state_dict.items():
        if 'weight' in name:
            # Add 1% gaussian noise
            noise = torch.randn_like(param) * 0.01 * param.std()
            state_dict[name] = param + noise
            print(f"Added noise to {name}")
    
    # Reset episode count to encourage fresh learning
    checkpoint['episode_count'] = 0
    checkpoint['step_count'] = 0
    
    # Save noisy checkpoint
    torch.save(checkpoint, "checkpoints/latest_checkpoint.pth")
    print("Saved noisy checkpoint to checkpoints/latest_checkpoint.pth")
    print("Use this checkpoint to resume training with more exploration")

if __name__ == "__main__":
    add_noise_to_checkpoint()