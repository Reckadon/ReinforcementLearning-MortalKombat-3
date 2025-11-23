#!/usr/bin/env python3
import time
import os
import torch
import diambra.arena
from diambra.arena import SpaceTypes, Roles, EnvironmentSettings

def load_trained_policy(checkpoint_path="checkpoints/latest_checkpoint.pth"):
    """
    Load the trained A2C policy from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        
    Returns:
        tuple: (model, device, frame_stack) - loaded model, device, and frame stack
    """
    # Dynamic import to avoid circular import issues
    import train_a2c
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create a temporary environment to get action space size
    temp_env = make_env()
    num_actions = temp_env.action_space.n
    temp_env.close()
    
    # Initialize model with the same configuration as training
    frame_shape = (128, 128)  # From environment settings
    num_features = 4  # character, health, aggressor_bar, timer
    stack_size = 4    # Frame stacking size
    
    model = train_a2c.ActorCritic(
        frame_shape=frame_shape,
        num_features=num_features,
        num_actions=num_actions,
        stack_size=stack_size
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set to evaluation mode
    
    # Initialize frame stack
    frame_stack = train_a2c.FrameStack(stack_size)
    
    print(f"Loaded trained policy from {checkpoint_path}")
    print(f"Action space size: {num_actions}")
    print(f"Using device: {device}")
    
    return model, device, frame_stack

def make_env():
    settings = EnvironmentSettings()
    settings.step_ratio = 4
    settings.frame_shape = (128, 128, 0)   # resized RGB frames
    # settings.frame_shape = (0, 0, 1)    # grayscale original size (example)
    # settings.frame_shape = (0, 0, 0)    # deactivated (original RGB)
    settings.action_space = SpaceTypes.DISCRETE     # or SpaceTypes.MULTI_DISCRETE
    settings.role = Roles.P1
    settings.continue_game = 0.0
    settings.show_final = False
    settings.difficulty = 1      # 1..5 for UMK3 - pick what you want
    settings.characters = "Kitana"
    env = diambra.arena.make("umk3", settings, render_mode="human")
    return env

def run_episode(env, model=None, device=None, frame_stack=None):
    obs, info = env.reset(seed=int(time.time() % 2**31))
    env.show_obs(obs)
    total_reward = 0.0
    steps = 0
    
    # Initialize frame stack for policy if using trained model
    if model is not None and frame_stack is not None:
        # Dynamic import to avoid circular import issues
        import train_a2c
        frame = train_a2c.preprocess_frame(obs["frame"]).to(device)
        frame_stack.reset(frame)

    while True:
        if model is not None and device is not None and frame_stack is not None:
            # Use trained policy
            import train_a2c
            frame = train_a2c.preprocess_frame(obs["frame"]).to(device)
            features = train_a2c.extract_features(obs).to(device)
            
            stacked_frames = frame_stack.get_stacked().unsqueeze(0)  # (1, stack_size, H, W)
            features_batch = features.unsqueeze(0)  # (1, num_features)
            
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(stacked_frames, features_batch)
                action = action.cpu().item()
            
            # Update frame stack for next iteration
            frame_stack.push(frame)
        else:
            # Fallback to random actions
            print("++"*20)
            action = env.action_space.sample()   # replace with your policy output
        
        # If using MULTI_DISCRETE you may get a list/tuple here; ensure your policy matches.
        # print("Action (policy):", action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        env.show_obs(obs)   # shows current frame (if render_mode supports it)

        done = terminated or truncated
        if done:
            return total_reward, steps, info

def main():
    print("🚀 Starting DIAMBRA Arena environment...")
    
    # Try to load trained policy first
    model = None
    device = None
    frame_stack = None
    
    try:
        model, device, frame_stack = load_trained_policy()
        print("🤖 Using trained A2C policy")
    except FileNotFoundError as e:
        print(f"⚠️  Warning: {e}")
        print("🎲 Falling back to random actions")
    except Exception as e:
        print(f"❌ Error loading policy: {e}")
        print("🎲 Falling back to random actions")
    
    # Initialize environment with better error handling
    env = None
    try:
        env = make_env()
        print("✅ DIAMBRA environment initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize DIAMBRA environment: {e}")
        print("💡 Make sure DIAMBRA Arena is properly installed and Docker is running")
        print("💡 You may need to run: diambra run python environment.py")
        return
    
    try:
        # Test environment reset
        print("🔄 Testing environment reset...")
        obs, info = env.reset()
        print(f"✅ Environment reset successful. Frame shape: {obs['frame'].shape}")
        
        # single episode example
        print("🎮 Running episode with trained policy...")
        total_reward, steps, info = run_episode(env, model, device, frame_stack)
        print(f"Episode finished — reward={total_reward:.2f}, steps={steps}")
        
        # Run a few more episodes to see the policy in action
        if model is not None:
            print("\n🎮 Running additional episodes with trained policy...")
            for i in range(2):  # Reduced to 2 additional episodes
                try:
                    total_reward, steps, info = run_episode(env, model, device, frame_stack)
                    print(f"Episode {i+2}: reward={total_reward:.2f}, steps={steps}")
                except Exception as e:
                    print(f"❌ Error in episode {i+2}: {e}")
                    break
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("💡 This might be a DIAMBRA connection issue. Try restarting the DIAMBRA container.")
    finally:
        if env is not None:
            try:
                env.close()
                print("🔚 Environment closed successfully")
            except Exception as e:
                print(f"⚠️  Warning: Error closing environment: {e}")

if __name__ == "__main__":
    main()
