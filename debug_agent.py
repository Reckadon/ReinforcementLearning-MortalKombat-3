#!/usr/bin/env python3
"""
Debug script to check training issues with A2C agent
"""

import torch
from environment import make_env
from train_a2c import preprocess_frame, extract_features, FrameStack, ActorCritic
import numpy as np

def debug_agent():
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("checkpoints/latest_checkpoint.pth", map_location=device)
    
    print("=== CHECKPOINT INFO ===----------------------------------------------")
    print(f"Training episode: {checkpoint.get('episode_count', 'unknown')}")
    print(f"Training steps: {checkpoint.get('step_count', 'unknown')}")
    
    # Check model weights
    state_dict = checkpoint['model_state_dict']
    print("\n=== MODEL WEIGHTS ANALYSIS ===")
    
    # Check if weights are reasonable (not all zeros or extreme values)
    for name, param in state_dict.items():
        if 'weight' in name:
            mean_val = param.abs().mean().item()
            max_val = param.abs().max().item()
            print(f"{name}: mean={mean_val:.6f}, max={max_val:.6f}")
    
    # Test environment setup
    print("\n=== ENVIRONMENT TEST ===")
    env = make_env()
    
    # Run a few random steps to see reward structure
    obs, info = env.reset()
    print(f"Initial obs keys: {obs.keys()}")
    print(f"P1 health: {obs['P1']['health']}")
    print(f"P2 health: {obs['P2']['health']}")
    
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i}: action={action}, reward={reward:.2f}, "
              f"P1_health={obs['P1']['health'][0]}, P2_health={obs['P2']['health'][0]}")
        
        if terminated or truncated:
            print(f"Episode ended early at step {i}")
            break
    
    print(f"Total reward from random actions: {total_reward}")
    env.close()
    
    # Test model predictions
    print("\n=== MODEL PREDICTION TEST ===")
    env = make_env()
    obs, _ = env.reset()
    
    # Get frame shape and create model
    frame_shape = obs["frame"].shape[:2]
    num_actions = env.action_space.n
    
    model = ActorCritic(
        frame_shape=frame_shape,
        num_features=4,
        num_actions=num_actions,
        stack_size=4
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Test a single prediction
    frame_stack = FrameStack(4)
    frame = preprocess_frame(obs["frame"]).to(device)
    frame_stack.reset(frame)
    features = extract_features(obs).to(device)
    
    stacked_frames = frame_stack.get_stacked().unsqueeze(0)
    features_batch = features.unsqueeze(0)
    
    with torch.no_grad():
        logits, value = model(stacked_frames, features_batch)
        probs = torch.softmax(logits, dim=1)
        
    print(f"Value prediction: {value.item():.3f}")
    print(f"Action probabilities: {probs.squeeze().cpu().numpy()}")
    print(f"Most likely action: {logits.argmax().item()}")
    print(f"Is output reasonable? Value should be roughly [-100, 100], probs should sum to 1.0")
    print(f"Probs sum: {probs.sum().item():.6f}")
    print("=== DEBUGGING COMPLETE ===------------------------------------------")
    
    env.close()

if __name__ == "__main__":
    debug_agent()