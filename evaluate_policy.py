#!/usr/bin/env python3
"""
Policy Evaluation Script for A2C-trained Ultimate Mortal Kombat 3 Agent

This script loads a trained A2C model checkpoint and evaluates it on the DIAMBRA UMK3 environment.
Supports deterministic and stochastic evaluation, video recording, and CSV result logging.

Usage Examples:
    # Basic evaluation with 20 episodes using deterministic policy
    python evaluate_policy.py --checkpoint checkpoints/latest_checkpoint.pth --n-episodes 20 --deterministic

    # Evaluation with video recording and CSV output
    python evaluate_policy.py --checkpoint checkpoints/latest_checkpoint.pth --n-episodes 5 --save-video ./videos --save-csv results.csv

    # Quick test with 3 episodes
    python evaluate_policy.py --checkpoint checkpoints/latest_checkpoint.pth --n-episodes 3
"""

import argparse
import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from environment import make_env

# Try to import from train_a2c.py, provide fallbacks if not available
try:
    from train_a2c import ActorCritic, preprocess_frame, extract_features, FrameStack
    IMPORTED_COMPONENTS = True
    print("Successfully imported components from train_a2c.py")
except Exception as e:
    print(f"Warning: Could not import from train_a2c.py: {e}")
    print("Using fallback implementations...")
    IMPORTED_COMPONENTS = False


# Fallback implementations if train_a2c.py is not available
if not IMPORTED_COMPONENTS:
    def preprocess_frame(frame):
        """
        Fallback: Convert RGB frame to grayscale and normalize to [0,1].

        Args:
            frame (np.ndarray): RGB frame of shape (H, W, 3)

        Returns:
            torch.Tensor: Grayscale frame of shape (1, H, W) in range [0,1]
        """
        gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        gray = gray.astype(np.float32) / 255.0
        gray_tensor = torch.from_numpy(gray).unsqueeze(0)  # (1, H, W)
        return gray_tensor

    def extract_features(obs):
        """
        Fallback: Extract and normalize RAM-like features from observation.

        Args:
            obs (dict): DIAMBRA observation dictionary

        Returns:
            torch.Tensor: Normalized features [character, health, aggressor_bar, timer]
        """
        p1 = obs["P1"]

        character = float(p1["character"]) / 25.0        # normalize [0..25] to [0,1]
        health = float(p1["health"][0]) / 166.0          # normalize [0..166] to [0,1]
        aggressor = float(p1["aggressor_bar"][0]) / 48.0 # normalize [0..48] to [0,1]
        timer = float(obs["timer"][0]) / 100.0           # normalize [0..100] to [0,1]

        features = torch.tensor([character, health, aggressor, timer], dtype=torch.float32)
        return features

    class FrameStack:
        """Fallback: Frame stacking utility using deque for efficient memory management."""

        def __init__(self, stack_size=4):
            self.stack_size = stack_size
            self.frames = deque(maxlen=stack_size)

        def reset(self, initial_frame):
            self.frames.clear()
            for _ in range(self.stack_size):
                self.frames.append(initial_frame.clone())

        def push(self, frame):
            self.frames.append(frame)

        def get_stacked(self):
            return torch.cat(list(self.frames), dim=0)

    class ActorCritic(nn.Module):
        """Fallback: Simplified Actor-Critic network."""

        def __init__(self, frame_shape, num_features, num_actions, stack_size=4):
            super(ActorCritic, self).__init__()

            self.frame_shape = frame_shape
            self.num_features = num_features
            self.num_actions = num_actions
            self.stack_size = stack_size

            # CNN encoder for stacked frames
            self.cnn = nn.Sequential(
                nn.Conv2d(stack_size, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten()
            )

            # Calculate CNN output size
            with torch.no_grad():
                dummy_input = torch.zeros(1, stack_size, *frame_shape)
                cnn_output_size = self.cnn(dummy_input).shape[1]

            # MLP encoder for RAM features
            self.feature_mlp = nn.Sequential(
                nn.Linear(num_features, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )

            # Combined embedding size
            combined_size = cnn_output_size + 32

            # Shared layer after concatenation
            self.shared = nn.Sequential(
                nn.Linear(combined_size, 256),
                nn.ReLU()
            )

            # Actor and critic heads
            self.actor = nn.Linear(256, num_actions)
            self.critic = nn.Linear(256, 1)

        def forward(self, frames, features):
            # CNN encoding
            cnn_features = self.cnn(frames)
            # MLP encoding
            mlp_features = self.feature_mlp(features)
            # Concatenate embeddings
            combined = torch.cat([cnn_features, mlp_features], dim=1)
            # Shared processing
            shared_features = self.shared(combined)
            # Actor and critic outputs
            action_logits = self.actor(shared_features)
            value = self.critic(shared_features)
            return action_logits, value

    # Minimal DQNNetwork fallback (to support DQN checkpoints)
    class DQNNetwork(nn.Module):
        def __init__(self, frame_shape, num_features, num_actions, stack_size=4):
            super(DQNNetwork, self).__init__()
            self.frame_shape = frame_shape
            self.num_features = num_features
            self.num_actions = num_actions
            self.stack_size = stack_size

            self.cnn = nn.Sequential(
                nn.Conv2d(stack_size, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten()
            )

            with torch.no_grad():
                dummy = torch.zeros(1, stack_size, *frame_shape)
                cnn_out = self.cnn(dummy).shape[1]

            self.feature_mlp = nn.Sequential(
                nn.Linear(num_features, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )

            combined_size = cnn_out + 32
            self.fc = nn.Sequential(
                nn.Linear(combined_size, 256),
                nn.ReLU()
            )

            self.q_head = nn.Linear(256, num_actions)

        def forward(self, frames, features):
            cnn_features = self.cnn(frames)
            mlp_features = self.feature_mlp(features)
            combined = torch.cat([cnn_features, mlp_features], dim=1)
            hidden = self.fc(combined)
            q_values = self.q_head(hidden)
            return q_values

        def act(self, frames, features, epsilon=0.0, device='cpu'):
            # frames: (stack_size, H, W) or (1, stack_size, H, W)
            # ensure batch dim
            if frames.dim() == 3:
                frames_b = frames.unsqueeze(0).to(device)
            else:
                frames_b = frames.to(device)
            features_b = features.to(device)
            with torch.no_grad():
                q_values = self.forward(frames_b, features_b)
                if random.random() < epsilon:
                    return int(torch.randint(0, q_values.shape[-1], (1,)).item()), q_values.squeeze(0)
                action = int(q_values.argmax(dim=-1).item())
            return action, q_values.squeeze(0)


def load_model(checkpoint_path, device):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path (str): Path to checkpoint file
        device (torch.device): Device to load model on

    Returns:
        tuple: (model, checkpoint_data, model_type) where model_type is 'a2c' or 'dqn'
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Case 1: A2C checkpoint with 'model_state_dict'
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']

        # Infer architecture parameters from state dict
        stack_size = 4  # default
        if 'cnn.0.weight' in state_dict:
            stack_size = state_dict['cnn.0.weight'].shape[1]

        num_actions = 12  # default
        if 'actor.weight' in state_dict:
            num_actions = state_dict['actor.weight'].shape[0]

        num_features = 4  # default
        if 'feature_mlp.0.weight' in state_dict:
            num_features = state_dict['feature_mlp.0.weight'].shape[1]

        print(f"Inferred A2C architecture: stack_size={stack_size}, num_actions={num_actions}, num_features={num_features}")

        env = make_env()
        obs, _ = env.reset()
        frame_shape = obs["frame"].shape[:2]  # (H, W)
        env.close()

        model = ActorCritic(
            frame_shape=frame_shape,
            num_features=num_features,
            num_actions=num_actions,
            stack_size=stack_size
        )

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, checkpoint, 'a2c'

    # Case 2: DQN checkpoint (online_state / target_state)
    elif 'online_state' in checkpoint:
        state_dict = checkpoint['online_state']

        # Try to infer params from state dict structure
        stack_size = 4
        if 'cnn.0.weight' in state_dict:
            stack_size = state_dict['cnn.0.weight'].shape[1]

        num_actions = 12
        # q_head weight probably named 'q_head.weight' or 'q_head.0.weight' depending on implementation
        if any(k.endswith('q_head.weight') or k.endswith('q_head.0.weight') or k.endswith('.q_head.weight') for k in state_dict.keys()):
            # naive search
            for k in state_dict.keys():
                if 'q_head' in k and 'weight' in k:
                    num_actions = state_dict[k].shape[0]
                    break
        else:
            # fallback: try to find any final linear layer weight with small out dim
            for k, v in state_dict.items():
                if v.ndim == 2 and v.shape[0] <= 100:  # heuristic
                    num_actions = v.shape[0]
                    break

        num_features = 4
        # try to find feature_mlp first weight naming
        if 'feature_mlp.0.weight' in state_dict:
            num_features = state_dict['feature_mlp.0.weight'].shape[1]
        else:
            # fallback heuristic: look for any linear with in_features equal to something small
            for k, v in state_dict.items():
                if v.ndim == 2 and v.shape[1] <= 128 and 'feature' in k:
                    num_features = v.shape[1]
                    break

        print(f"Inferred DQN architecture: stack_size={stack_size}, num_actions={num_actions}, num_features={num_features}")

        env = make_env()
        obs, _ = env.reset()
        frame_shape = obs["frame"].shape[:2]
        env.close()

        # attempt to import DQNNetwork from train_dqn or DQN module if available
        try:
            from train_dqn import DQNNetwork as ImportedDQNNetwork  # if you have a separate file
            DQNClass = ImportedDQNNetwork
        except Exception:
            # fallback to our local DQNNetwork defined above (if present)
            try:
                DQNClass = DQNNetwork
            except NameError:
                raise RuntimeError("DQNNetwork class not available to construct model")

        model = DQNClass(
            frame_shape=frame_shape,
            num_features=num_features,
            num_actions=num_actions,
            stack_size=stack_size
        )

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, checkpoint, 'dqn'

    else:
        raise ValueError("Checkpoint does not contain 'model_state_dict' (A2C) or 'online_state' (DQN) keys")


def build_state(obs, frame_stack, device):
    """
    Build state tensors from observation.

    Args:
        obs (dict): DIAMBRA observation
        frame_stack (FrameStack): Frame stacking utility
        device (torch.device): Device for tensors

    Returns:
        tuple: (stacked_frames, features)
    """
    # Process frame and update stack
    frame = preprocess_frame(obs["frame"]).to(device)
    frame_stack.push(frame)

    # Get stacked frames and features
    stacked_frames = frame_stack.get_stacked().unsqueeze(0)  # (1, stack_size, H, W)
    features = extract_features(obs).to(device).unsqueeze(0)  # (1, num_features)

    return stacked_frames, features


def save_frame_as_image(frame, filepath):
    """
    Save frame as PNG image.

    Args:
        frame (np.ndarray): RGB frame of shape (H, W, 3)
        filepath (str): Output file path
    """
    try:
        from PIL import Image
        img = Image.fromarray(frame.astype(np.uint8))
        img.save(filepath)
    except ImportError:
        # Fallback to opencv if PIL not available
        try:
            import cv2
            cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        except ImportError:
            print("Warning: Neither PIL nor OpenCV available for image saving")


import random  # used by DQN stochastic sampling


def run_episode(env, model, model_type, device, deterministic=False, stack_size=4, max_steps=None, save_video_path=None, episode_num=0):
    """
    Run a single evaluation episode.

    Args:
        env: DIAMBRA environment
        model: Trained model (ActorCritic or DQNNetwork)
        model_type (str): 'a2c' or 'dqn'
        device: PyTorch device
        deterministic (bool): Whether to use deterministic policy
        stack_size (int): Number of frames to stack
        max_steps (int): Maximum steps per episode
        save_video_path (str): Path to save video frames
        episode_num (int): Episode number for video naming

    Returns:
        tuple: (episode_reward, episode_length, episode_info)
    """
    obs, info = env.reset()

    # Initialize frame stack
    frame_stack = FrameStack(stack_size)
    initial_frame = preprocess_frame(obs["frame"]).to(device)
    frame_stack.reset(initial_frame)

    episode_reward = 0.0
    episode_length = 0
    episode_info = {}

    # Setup video saving if requested
    if save_video_path:
        episode_video_dir = os.path.join(save_video_path, f"episode_{episode_num:03d}")
        os.makedirs(episode_video_dir, exist_ok=True)
        frame_count = 0

    while True:
        # Build state
        stacked_frames, features = build_state(obs, frame_stack, device)

        # Get action from model depending on model type
        with torch.no_grad():
            if model_type == 'a2c':
                action_logits, value = model(stacked_frames, features)

                if deterministic:
                    action = action_logits.argmax(dim=1).item()
                else:
                    action_dist = torch.distributions.Categorical(logits=action_logits)
                    action = action_dist.sample().item()

            elif model_type == 'dqn':
                # DQN: forward returns Q-values
                # stacked_frames shape: (1, stack, H, W), features shape (1, num_feat)
                q_values = model(stacked_frames, features)
                if deterministic:
                    action = int(q_values.argmax(dim=1).item())
                else:
                    # stochastic mode for DQN: small epsilon-greedy (keeps behavior similar to A2C's stochasticity)
                    eps = 0.1
                    if random.random() < eps:
                        action = random.randrange(q_values.shape[-1])
                    else:
                        action = int(q_values.argmax(dim=1).item())
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

        # Save frame if video recording is enabled
        if save_video_path:
            frame_path = os.path.join(episode_video_dir, f"frame_{frame_count:06d}.png")
            save_frame_as_image(obs["frame"], frame_path)
            frame_count += 1

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        episode_length += 1

        # Check termination conditions
        done = terminated or truncated
        if max_steps and episode_length >= max_steps:
            done = True

        if done:
            # Extract additional info if available
            episode_info = {
                'terminated': terminated,
                'truncated': truncated,
                'final_info': info
            }
            break

    return episode_reward, episode_length, episode_info


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained A2C policy on UMK3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--env-seed", type=int, default=0, help="Environment seed")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--stack-size", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy (argmax)")
    parser.add_argument("--render", action="store_true", default=True, help="Render environment")
    parser.add_argument("--save-video", help="Directory to save episode videos as PNG frames")
    parser.add_argument("--save-csv", help="Path to save results as CSV")
    parser.add_argument("--device", help="Device to use (cpu/cuda, default: auto)")
    parser.add_argument("--max-steps-per-ep", type=int, help="Maximum steps per episode")

    args = parser.parse_args()

    # Auto-detect device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directories
    if args.save_video:
        os.makedirs(args.save_video, exist_ok=True)
        print(f"Video frames will be saved to: {args.save_video}")

    try:
        # Load model
        model, checkpoint, model_type = load_model(args.checkpoint, device)
        print(f"Model loaded successfully (type: {model_type})")
        if 'episode_count' in checkpoint:
            print(f"Checkpoint from training episode: {checkpoint['episode_count']}")

        # Create environment
        env = make_env()
        env_name = 'UMK3'
        if hasattr(env, 'spec') and env.spec is not None:
            env_name = env.spec.id
        print(f"Environment created: {env_name}")
        print(f"Action space: {env.action_space}")

        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        episode_data = []

        print(f"\nStarting evaluation with {args.n_episodes} episodes...")
        print(f"Policy mode: {'Deterministic' if args.deterministic else 'Stochastic'}")
        print("-" * 60)

        start_time = time.time()

        for episode in range(args.n_episodes):
            try:
                episode_reward, episode_length, episode_info = run_episode(
                    env=env,
                    model=model,
                    model_type=model_type,
                    device=device,
                    deterministic=args.deterministic,
                    stack_size=args.stack_size,
                    max_steps=args.max_steps_per_ep,
                    save_video_path=args.save_video,
                    episode_num=episode
                )

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                # Store episode data for CSV
                episode_data.append({
                    'episode': episode,
                    'reward': episode_reward,
                    'length': episode_length,
                    'deterministic': args.deterministic,
                    'seed': args.env_seed,
                    'terminated': episode_info.get('terminated', False),
                    'truncated': episode_info.get('truncated', False)
                })

                # Print progress
                print(f"Episode {episode + 1:3d}/{args.n_episodes}: "
                      f"Reward = {episode_reward:8.2f}, "
                      f"Length = {episode_length:4d} steps")

            except KeyboardInterrupt:
                print(f"\nInterrupted during episode {episode + 1}")
                break
            except Exception as e:
                print(f"Error in episode {episode + 1}: {e}")
                continue

        # Calculate and display results
        if episode_rewards:
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            std_length = np.std(episode_lengths)

            elapsed_time = time.time() - start_time

            print("-" * 60)
            print("EVALUATION RESULTS")
            print("-" * 60)
            print(f"Episodes completed: {len(episode_rewards)}")
            print(f"Total time: {elapsed_time:.2f} seconds")
            print(f"Average episode time: {elapsed_time / len(episode_rewards):.2f} seconds")
            print()
            print(f"Reward Statistics:")
            print(f"  Mean: {mean_reward:.3f} ± {std_reward:.3f}")
            print(f"  Min:  {min(episode_rewards):.3f}")
            print(f"  Max:  {max(episode_rewards):.3f}")
            print()
            print(f"Episode Length Statistics:")
            print(f"  Mean: {mean_length:.1f} ± {std_length:.1f}")
            print(f"  Min:  {min(episode_lengths)}")
            print(f"  Max:  {max(episode_lengths)}")

            # Save CSV if requested
            if args.save_csv and episode_data:
                with open(args.save_csv, 'w', newline='') as csvfile:
                    fieldnames = ['episode', 'reward', 'length', 'deterministic', 'seed', 'terminated', 'truncated']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in episode_data:
                        writer.writerow(row)
                print(f"\nResults saved to: {args.save_csv}")

        else:
            print("No episodes completed successfully")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
            print("\nEnvironment closed")
        except:
            pass


if __name__ == "__main__":
    main()
