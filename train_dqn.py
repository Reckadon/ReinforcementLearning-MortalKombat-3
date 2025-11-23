"""
DQN (Deep Q-Network) implementation for Ultimate Mortal Kombat 3
Adapted with minimal changes from the provided A2C script.

Key changes (see separate explanation below):
 - Replace ActorCritic with DQNNetwork (single Q-head outputting Q-values for each action)
 - Add ReplayBuffer for off-policy learning
 - Use epsilon-greedy action selection and an epsilon schedule
 - Use target network for stable Q-learning and periodic soft/hard updates
 - Loss: MSE between target Q-values and online Q-values (no actor/critic/entropy)
 - Training is step-based (env steps -> buffer -> train every train_freq steps)

This file is structured to keep your original utilities (preprocess_frame, extract_features,
FrameStack, environment setup) intact so changes are minimal and easy to audit.
"""

import os
import time
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from environment import make_env

# ---------------------- utilities carried over (unchanged) ----------------------

def preprocess_frame(frame):
    gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
    gray = gray.astype(np.float32) / 255.0
    gray_tensor = torch.from_numpy(gray).unsqueeze(0)  # (1, H, W)
    return gray_tensor


def extract_features(obs):
    p1 = obs["P1"]
    character = float(p1["character"])        # int in [0..25]
    health = float(p1["health"][0])          # int in [0..166]
    aggressor = float(p1["aggressor_bar"][0]) # int in [0..48]
    timer = float(obs["timer"][0])           # int in [0..100]

    norm_character = character / 25.0
    norm_health = health / 166.0
    norm_aggressor = aggressor / 48.0
    norm_timer = timer / 100.0

    features = torch.tensor([norm_character, norm_health, norm_aggressor, norm_timer],
                            dtype=torch.float32)
    return features


class FrameStack:
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


# ---------------------- DQN network (replaces ActorCritic) ----------------------
class DQNNetwork(nn.Module):
    def __init__(self, frame_shape, num_features, num_actions, stack_size=4):
        super(DQNNetwork, self).__init__()
        self.frame_shape = frame_shape
        self.num_features = num_features
        self.num_actions = num_actions
        self.stack_size = stack_size

        # CNN encoder for stacked frames (same conv layout as A2C)
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

        # Q-value head (one value per action)
        self.q_head = nn.Linear(256, num_actions)

    def forward(self, frames, features):
        # frames: (batch, stack_size, H, W)
        cnn_features = self.cnn(frames)
        mlp_features = self.feature_mlp(features)
        combined = torch.cat([cnn_features, mlp_features], dim=1)
        hidden = self.fc(combined)
        q_values = self.q_head(hidden)
        return q_values

    def act(self, frames, features, epsilon=0.0, device='cpu'):
        """Epsilon-greedy action selection. frames/features are single-step tensors (no batch).
        Returns: action (int) and q_values (torch.Tensor)
        """
        if random.random() < epsilon:
            return random.randrange(self.q_head.out_features), None
        frames = frames.unsqueeze(0).to(device)
        features = features.unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.forward(frames, features)
            action = int(q_values.argmax(dim=-1).item())
        return action, q_values.squeeze(0)


# ---------------------- Replay buffer ----------------------
Transition = namedtuple('Transition', ('state_frames', 'state_features', 'action', 'reward', 'next_frames', 'next_features', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # convert to tensors
        state_frames = torch.stack([b.state_frames for b in batch])
        state_features = torch.stack([b.state_features for b in batch])
        actions = torch.tensor([b.action for b in batch], dtype=torch.long)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32)
        next_frames = torch.stack([b.next_frames for b in batch])
        next_features = torch.stack([b.next_features for b in batch])
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32)
        return state_frames, state_features, actions, rewards, next_frames, next_features, dones

    def __len__(self):
        return len(self.buffer)


# ---------------------- Trainer adapted for DQN ----------------------
class DQNTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.env = make_env()
        obs, _ = self.env.reset()
        frame_shape = obs['frame'].shape[:2]
        num_actions = self.env.action_space.n
        num_features = 4

        self.frame_stack = FrameStack(config['stack_size'])

        self.online_net = DQNNetwork(frame_shape, num_features, num_actions, stack_size=config['stack_size']).to(self.device)
        self.target_net = DQNNetwork(frame_shape, num_features, num_actions, stack_size=config['stack_size']).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config['learning_rate'])

        self.replay = ReplayBuffer(config['replay_capacity'])

        self.writer = SummaryWriter(config['log_dir'])

        self.epsilon_start = config['epsilon_start']
        self.epsilon_final = config['epsilon_final']
        self.epsilon_decay = config['epsilon_decay']

        self.step_count = 0
        self.episode_count = 0

        self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.config['save_dir'], 'dqn_latest.pth')
        if os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                self.online_net.load_state_dict(ckpt['online_state'])
                self.target_net.load_state_dict(ckpt.get('target_state', ckpt['online_state']))
                # Load optimizer state and move its tensors to the trainer device
                if 'opt' in ckpt and ckpt['opt'] is not None:
                    self.optimizer.load_state_dict(ckpt['opt'])
                    # Ensure optimizer state tensors are on the correct device
                    for state in self.optimizer.state.values():
                        for k, v in list(state.items()):
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                self.step_count = ckpt.get('step_count', 0)
                self.episode_count = ckpt.get('episode_count', 0)
                print(f"Loaded DQN checkpoint at episode {self.episode_count}, step {self.step_count}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")

    def save_checkpoint(self):
        os.makedirs(self.config['save_dir'], exist_ok=True)
        checkpoint_path = os.path.join(self.config['save_dir'], 'dqn_latest.pth')
        torch.save({
            'online_state': self.online_net.state_dict(),
            'target_state': self.target_net.state_dict(),
            'opt': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }, checkpoint_path)
        # print(f"Saved DQN checkpoint at episode {self.episode_count}")

    def epsilon_by_frame(self, frame_idx):
        # Linear decay
        return max(self.epsilon_final, self.epsilon_start - frame_idx / self.epsilon_decay)

    def compute_td_loss(self, batch_size):
        state_frames, state_features, actions, rewards, next_frames, next_features, dones = self.replay.sample(batch_size)
        state_frames = state_frames.to(self.device)
        state_features = state_features.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_frames = next_frames.to(self.device)
        next_features = next_features.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values
        q_values = self.online_net(state_frames, state_features)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q-values (Double DQN could be added; here simple DQN using target net)
        with torch.no_grad():
            next_q_values = self.target_net(next_frames, next_features)
            next_q_value = next_q_values.max(1)[0]
            expected_q = rewards + (1 - dones) * self.config['gamma'] * next_q_value

        loss = F.mse_loss(q_value, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config['max_grad_norm'])
        self.optimizer.step()

        return loss.item(), q_value.mean().item()

    def train(self):
        print("Starting DQN training...")
        last_save_time = time.time()

        obs, _ = self.env.reset()
        frame = preprocess_frame(obs['frame']).to(self.device)
        features = extract_features(obs).to(self.device)
        self.frame_stack.reset(frame)
        episode_reward = 0
        episode_length = 0

        try:
            while self.episode_count < self.config['max_episodes']:
                epsilon = self.epsilon_by_frame(self.step_count)
                stacked_frames = self.frame_stack.get_stacked().unsqueeze(0).squeeze(0)  # (stack, H, W)

                action, _ = self.online_net.act(stacked_frames, features, epsilon=epsilon, device=self.device)

                # Step env
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_frame = preprocess_frame(obs['frame']).to(self.device)
                next_features = extract_features(obs).to(self.device)

                # If done, next stacked frames should be reset copies
                if done:
                    # next stacked frames: for terminal we've been resetting to next episode start below
                    next_stacked = stacked_frames.clone()
                else:
                    # push and get next stack
                    self.frame_stack.push(next_frame)
                    next_stacked = self.frame_stack.get_stacked()

                # store transition (store tensors on CPU to save GPU memory)
                self.replay.push(stacked_frames.cpu(), features.cpu(), action, reward, next_stacked.cpu(), next_features.cpu(), done)

                episode_reward += reward
                episode_length += 1
                self.step_count += 1

                # training step
                if len(self.replay) >= self.config['batch_size'] and self.step_count % self.config['train_freq'] == 0:
                    loss, avg_q = self.compute_td_loss(self.config['batch_size'])
                    self.writer.add_scalar('Loss/TD', loss, self.step_count)
                    self.writer.add_scalar('Train/AvgQ', avg_q, self.step_count)

                # update target network
                if self.step_count % self.config['target_update'] == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                if done:
                    # log episode
                    if self.episode_count % self.config['eval_interval'] == 0:
                        # Run evaluation episode (greedy)
                        eval_reward, eval_len = self.evaluate_episode()
                        self.writer.add_scalar('Evaluation/Reward', eval_reward, self.episode_count)
                        self.writer.add_scalar('Evaluation/Length', eval_len, self.episode_count)
                        print(f"Episode {self.episode_count}: Eval Reward = {eval_reward:.2f}, Length = {eval_len}")

                    if self.episode_count % self.config['log_interval'] == 0:
                        print(f"Episode {self.episode_count}: Step {self.step_count}, Ep Reward = {episode_reward:.2f}, Ep Len = {episode_length}, Epsilon = {epsilon:.3f}")
                        self.writer.add_scalar('Episode/Reward', episode_reward, self.episode_count)
                        self.writer.add_scalar('Episode/Length', episode_length, self.episode_count)

                    # reset env
                    obs, _ = self.env.reset()
                    frame = preprocess_frame(obs['frame']).to(self.device)
                    features = extract_features(obs).to(self.device)
                    self.frame_stack.reset(frame)
                    episode_reward = 0
                    episode_length = 0
                    self.episode_count += 1
                else:
                    # continue with next state
                    frame = next_frame
                    features = next_features

                # periodic save
                current_time = time.time()
                if (current_time - last_save_time >= self.config['save_interval'] or
                    self.episode_count % self.config['save_interval'] == 0):
                    self.save_checkpoint()
                    last_save_time = current_time

        except KeyboardInterrupt:
            print('\nTraining interrupted by user')
        finally:
            self.save_checkpoint()
            self.writer.close()
            self.env.close()
            print('Training finished and resources cleaned up')

    def evaluate_episode(self):
        obs, _ = self.env.reset()
        frame = preprocess_frame(obs['frame']).to(self.device)
        features = extract_features(obs).to(self.device)
        self.frame_stack.reset(frame)

        episode_reward = 0
        episode_length = 0
        while True:
            stacked_frames = self.frame_stack.get_stacked().unsqueeze(0).squeeze(0)
            action, _ = self.online_net.act(stacked_frames, features, epsilon=0.0, device=self.device)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            if done:
                break
            frame = preprocess_frame(obs['frame']).to(self.device)
            features = extract_features(obs).to(self.device)
            self.frame_stack.push(frame)
        return episode_reward, episode_length


# ---------------------- main ----------------------
def main():
    config = {
        'stack_size': 4,
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'max_grad_norm': 0.5,

        # Replay and training
        'replay_capacity': 100000,
        'batch_size': 64,
        'train_freq': 4,          # train every 4 steps
        'target_update': 1000,    # hard update frequency in steps

        # Epsilon schedule
        'epsilon_start': 1.0,
        'epsilon_final': 0.01,
        'epsilon_decay': 100000,  # linear decay across steps

        # Training
        'max_episodes': 10000,

        # Logging and saving
        'log_interval': 10,
        'eval_interval': 50,
        'save_interval': 300,  # seconds
        'log_dir': 'runs/dqn_umk3',
        'save_dir': 'checkpoints_dqn'
    }

    trainer = DQNTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
