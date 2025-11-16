#!/usr/bin/env python3
"""
A2C (Advantage Actor-Critic) Training Script for Ultimate Mortal Kombat 3
Using PyTorch and DIAMBRA Arena Environment

This script implements synchronous A2C with:
- CNN encoder for grayscale frames with frame stacking
- MLP encoder for RAM features (character, health, aggressor_bar, timer)  
- Combined actor-critic architecture
- TensorBoard logging and model checkpointing
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from environment import make_env


def preprocess_frame(frame):
    """
    Convert RGB frame to grayscale and normalize to [0,1].
    
    Args:
        frame (np.ndarray): RGB frame of shape (H, W, 3)
        
    Returns:
        torch.Tensor: Grayscale frame of shape (1, H, W) in range [0,1]
    """
    # Convert to grayscale using standard weights
    gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
    # Normalize to [0,1]
    gray = gray.astype(np.float32) / 255.0
    # Convert to torch tensor and add channel dimension
    gray_tensor = torch.from_numpy(gray).unsqueeze(0)  # (1, H, W)
    return gray_tensor


def extract_features(obs):
    """
    Extract and normalize RAM-like features from observation.
    
    Args:
        obs (dict): DIAMBRA observation dictionary
        
    Returns:
        torch.Tensor: Normalized features [character, health, aggressor_bar, timer]
    """
    p1 = obs["P1"]
    
    # Extract raw features
    character = float(p1["character"])        # int in [0..25]
    health = float(p1["health"][0])          # int in [0..166]
    aggressor = float(p1["aggressor_bar"][0]) # int in [0..48]
    timer = float(obs["timer"][0])           # int in [0..100]
    
    # Normalize to [0,1]
    norm_character = character / 25.0
    norm_health = health / 166.0
    norm_aggressor = aggressor / 48.0
    norm_timer = timer / 100.0
    
    features = torch.tensor([norm_character, norm_health, norm_aggressor, norm_timer], 
                           dtype=torch.float32)
    return features


class FrameStack:
    """Frame stacking utility using deque for efficient memory management."""
    
    def __init__(self, stack_size=4):
        """
        Initialize frame stack.
        
        Args:
            stack_size (int): Number of frames to stack
        """
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        
    def reset(self, initial_frame):
        """
        Reset the frame stack with initial frame.
        
        Args:
            initial_frame (torch.Tensor): Initial frame of shape (1, H, W)
        """
        self.frames.clear()
        # Fill with copies of the initial frame
        for _ in range(self.stack_size):
            self.frames.append(initial_frame.clone())
    
    def push(self, frame):
        """
        Add new frame to stack.
        
        Args:
            frame (torch.Tensor): New frame of shape (1, H, W)
        """
        self.frames.append(frame)
    
    def get_stacked(self):
        """
        Get stacked frames.
        Returns:
            torch.Tensor: Stacked frames of shape (stack_size, H, W)
        """
        return torch.cat(list(self.frames), dim=0)


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network with CNN encoder for frames and MLP for features.
    """
    
    def __init__(self, frame_shape, num_features, num_actions, stack_size=4):
        """
        Initialize Actor-Critic network.
        
        Args:
            frame_shape (tuple): Shape of single frame (H, W)
            num_features (int): Number of RAM features
            num_actions (int): Number of discrete actions
            stack_size (int): Number of stacked frames
        """
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
        
        # Actor head (policy)
        self.actor = nn.Linear(256, num_actions)
        
        # Critic head (value function)
        self.critic = nn.Linear(256, 1)
        
    def forward(self, frames, features):
        """
        Forward pass through the network.
        
        Args:
            frames (torch.Tensor): Stacked frames of shape (batch_size, stack_size, H, W)
            features (torch.Tensor): RAM features of shape (batch_size, num_features)
            
        Returns:
            tuple: (action_logits, value)
        """
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
    
    def get_action_and_value(self, frames, features, action=None):
        """
        Get action distribution, sampled action, log probability, and value.
        
        Args:
            frames (torch.Tensor): Stacked frames
            features (torch.Tensor): RAM features  
            action (torch.Tensor, optional): Specific action to evaluate
            
        Returns:
            tuple: (action, log_prob, entropy, value)
        """
        action_logits, value = self.forward(frames, features)
        
        # Apply temperature scaling to prevent overconfident predictions
        temperature = 1.0
        scaled_logits = action_logits / temperature
        
        action_dist = torch.distributions.Categorical(logits=scaled_logits)
        
        if action is None:
            action = action_dist.sample()
        
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        return action, log_prob, entropy, value.squeeze()


def compute_returns_and_advantages(rewards, values, next_value, dones, gamma=0.99):
    """
    Compute returns and advantages using GAE.
    
    Args:
        rewards (list): List of rewards
        values (list): List of value estimates
        next_value (float): Value estimate for next state
        dones (list): List of done flags
        gamma (float): Discount factor
        
    Returns:
        tuple: (returns, advantages)
    """
    returns = []
    advantages = []
    
    # Compute returns
    R = next_value
    for i in reversed(range(len(rewards))):
        R = rewards[i] + gamma * R * (1 - dones[i])
        returns.insert(0, R)
    
    # Compute advantages
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    values_tensor = torch.stack(values)
    advantages_tensor = returns_tensor - values_tensor
    
    return returns_tensor, advantages_tensor


class A2CTrainer:
    """Main A2C training class."""
    
    def __init__(self, config):
        """
        Initialize A2C trainer.
        
        Args:
            config (dict): Training configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Environment
        self.env = make_env()
        
        # Get environment information
        obs, _ = self.env.reset()
        frame_shape = obs["frame"].shape[:2]  # (H, W)
        num_actions = self.env.action_space.n
        num_features = 4  # character, health, aggressor_bar, timer
        
        print(f"Frame shape: {frame_shape}")
        print(f"Number of actions: {num_actions}")
        print(f"Number of features: {num_features}")
        
        # Frame stacking
        self.frame_stack = FrameStack(config["stack_size"])
        
        # Model
        self.model = ActorCritic(
            frame_shape=frame_shape,
            num_features=num_features,
            num_actions=num_actions,
            stack_size=config["stack_size"]
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        
        # Logging
        self.writer = SummaryWriter(config["log_dir"])
        
        # Training state
        self.episode_count = 0
        self.step_count = 0
        
        # Load checkpoint if available
        self.load_checkpoint()
        
    def load_checkpoint(self):
        """Load model checkpoint if available."""
        checkpoint_path = os.path.join(self.config["save_dir"], "latest_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.episode_count = checkpoint["episode_count"]
                self.step_count = checkpoint["step_count"]
                print(f"Loaded checkpoint from episode {self.episode_count}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
        
    def save_checkpoint(self):
        """Save model checkpoint."""
        os.makedirs(self.config["save_dir"], exist_ok=True)
        checkpoint_path = os.path.join(self.config["save_dir"], "latest_checkpoint.pth")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_count": self.episode_count,
            "step_count": self.step_count
        }, checkpoint_path)
        print(f"Saved checkpoint at episode {self.episode_count}")
        
    def rollout(self, n_steps):
        """
        Collect rollout data.
        
        Args:
            n_steps (int): Number of steps to collect
            
        Returns:
            dict: Rollout data
        """
        observations = []
        frames_list = []
        features_list = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []
        
        obs, _ = self.env.reset()
        frame = preprocess_frame(obs["frame"]).to(self.device)
        features = extract_features(obs).to(self.device)
        self.frame_stack.reset(frame)
        
        for step in range(n_steps):
            stacked_frames = self.frame_stack.get_stacked().unsqueeze(0)  # (1, stack_size, H, W)
            features_batch = features.unsqueeze(0)  # (1, num_features)
            
            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action_and_value(
                    stacked_frames, features_batch
                )
            
            # Store data
            frames_list.append(stacked_frames.squeeze(0))
            features_list.append(features)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            
            # Step environment
            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().item())
            done = terminated or truncated
            
            rewards.append(reward)
            dones.append(done)
            
            if done:
                obs, _ = self.env.reset()
                frame = preprocess_frame(obs["frame"]).to(self.device)
                features = extract_features(obs).to(self.device)
                self.frame_stack.reset(frame)
            else:
                frame = preprocess_frame(obs["frame"]).to(self.device)
                features = extract_features(obs).to(self.device)
                self.frame_stack.push(frame)
        
        # Get final value for bootstrap
        stacked_frames = self.frame_stack.get_stacked().unsqueeze(0)
        features_batch = features.unsqueeze(0)
        with torch.no_grad():
            _, _, _, next_value = self.model.get_action_and_value(stacked_frames, features_batch)
        
        return {
            "frames": torch.stack(frames_list),
            "features": torch.stack(features_list),
            "actions": torch.stack(actions),
            "rewards": rewards,
            "log_probs": torch.stack(log_probs),
            "values": values,
            "dones": dones,
            "next_value": next_value.item()
        }
    
    def update(self, rollout_data):
        """
        Update the model using rollout data.
        
        Args:
            rollout_data (dict): Data from rollout
        """
        # Compute returns and advantages
        returns, advantages = compute_returns_and_advantages(
            rollout_data["rewards"],
            rollout_data["values"],
            rollout_data["next_value"],
            rollout_data["dones"],
            gamma=self.config["gamma"]
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to device
        frames = rollout_data["frames"].to(self.device)
        features = rollout_data["features"].to(self.device)
        actions = rollout_data["actions"].to(self.device)
        old_log_probs = rollout_data["log_probs"].to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        
        # Forward pass
        _, log_probs, entropy, values = self.model.get_action_and_value(
            frames, features, actions
        )
        
        # Compute losses
        # Actor loss: -log_prob * advantage
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic loss: MSE between predicted and actual returns
        critic_loss = F.mse_loss(values, returns)
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (
            actor_loss +
            self.config["value_loss_coef"] * critic_loss +
            self.config["entropy_coef"] * entropy_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
        
        self.optimizer.step()
        
        # Log metrics
        self.writer.add_scalar("Losses/Actor", actor_loss.item(), self.step_count)
        self.writer.add_scalar("Losses/Critic", critic_loss.item(), self.step_count)
        self.writer.add_scalar("Losses/Entropy", entropy_loss.item(), self.step_count)
        self.writer.add_scalar("Losses/Total", total_loss.item(), self.step_count)
        self.writer.add_scalar("Training/Entropy", entropy.mean().item(), self.step_count)
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "entropy": entropy.mean().item()
        }
    
    def evaluate_episode(self):
        """Run one evaluation episode and return metrics."""
        obs, _ = self.env.reset()
        frame = preprocess_frame(obs["frame"]).to(self.device)
        features = extract_features(obs).to(self.device)
        self.frame_stack.reset(frame)
        
        episode_reward = 0
        episode_length = 0
        
        while True:
            stacked_frames = self.frame_stack.get_stacked().unsqueeze(0)
            features_batch = features.unsqueeze(0)
            
            with torch.no_grad():
                action, _, _, _ = self.model.get_action_and_value(stacked_frames, features_batch)
            
            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().item())
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
                
            frame = preprocess_frame(obs["frame"]).to(self.device)
            features = extract_features(obs).to(self.device)
            self.frame_stack.push(frame)
        
        return episode_reward, episode_length
    
    def train(self):
        """Main training loop."""
        print("Starting A2C training...")
        last_save_time = time.time()
        
        try:
            while self.episode_count < self.config["max_episodes"]:
                # Collect rollout
                rollout_data = self.rollout(self.config["n_steps"])
                
                # Update model
                loss_info = self.update(rollout_data)
                
                # Update step count
                self.step_count += self.config["n_steps"]
                
                # Evaluate every eval_interval episodes
                if self.episode_count % self.config["eval_interval"] == 0:
                    eval_reward, eval_length = self.evaluate_episode()
                    self.writer.add_scalar("Evaluation/Reward", eval_reward, self.episode_count)
                    self.writer.add_scalar("Evaluation/Length", eval_length, self.episode_count)
                    print(f"Episode {self.episode_count}: Eval Reward = {eval_reward:.2f}, Length = {eval_length}")
                
                # Log training info
                if self.episode_count % self.config["log_interval"] == 0:
                    print(f"Episode {self.episode_count}: "
                          f"Actor Loss = {loss_info['actor_loss']:.4f}, "
                          f"Critic Loss = {loss_info['critic_loss']:.4f}, "
                          f"Entropy = {loss_info['entropy']:.4f}")
                
                # Save checkpoint
                current_time = time.time()
                if (current_time - last_save_time >= self.config["save_interval"] or 
                    self.episode_count % self.config["save_interval"] == 0):
                    self.save_checkpoint()
                    last_save_time = current_time
                
                self.episode_count += 1
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            self.save_checkpoint()
            self.writer.close()
            self.env.close()
            print("Training finished and resources cleaned up")


def main():
    """Main function to start training."""
    # Training configuration
    config = {
        # Environment
        "stack_size": 4,
        
        # Model
        "learning_rate": 1e-4,  # Reduced from 3e-4 for more stable learning
        "gamma": 0.99,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.1,  # Increased from 0.01 to maintain exploration
        "max_grad_norm": 0.5,
        
        # Training
        "n_steps": 128,
        "max_episodes": 10000,
        
        # Logging and saving
        "log_interval": 10,
        "eval_interval": 50,
        "save_interval": 100,  # episodes
        "log_dir": "runs/a2c_umk3",
        "save_dir": "checkpoints"
    }
    
    # Create trainer and start training
    trainer = A2CTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()