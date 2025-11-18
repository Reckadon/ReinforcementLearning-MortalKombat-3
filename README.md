# ppo Agent for Ultimate Mortal Kombat 3

This branch contains an Advantage Actor-Critic (ppo) reinforcement learning implementation for playing Ultimate Mortal Kombat 3 using the DIAMBRA Arena environment.

## Architecture Overview

### Neural Network Architecture
- **Dual-Input Design**: Combines visual frame data with game state features
- **CNN Encoder**: Processes stacked grayscale frames using 3-layer convolutional network
- **MLP Encoder**: Handles normalized game state features (character, health, aggressor bar, timer)
- **Actor-Critic Head**: Shared representation feeding into policy (actor) and value function (critic) outputs

### Key Features
- **Frame Preprocessing**: RGB frames converted to grayscale and normalized
- **Frame Stacking**: Uses 4-frame history for temporal information
- **Feature Extraction**: Normalizes game state variables to [0,1] range
- **Temperature Scaling**: Prevents overconfident action predictions

## File Structure

### Core Training Files
- `train_ppo.py` - Main ppo training script with complete implementation
- `environment.py` - DIAMBRA environment setup and configuration
- `evaluate_policy.py` - Policy evaluation script with deterministic/stochastic modes

### Utility Scripts
- `debug_agent.py` - Debugging script for analyzing model weights and training issues
- `add_noise.py` - Adds Gaussian noise to model weights to escape local minima

### Training Outputs
- `checkpoints/` - Model checkpoints and training state
- `runs/` - TensorBoard logging data
- `results.csv` - Evaluation results storage

## Input Processing

### Frame Processing
- Converts 128x128x3 RGB frames to grayscale using standard luminance weights
- Normalizes pixel values to [0,1] range
- Stacks 4 consecutive frames for temporal context
- Input shape: (batch_size, 4, 128, 128)

### Feature Processing
Extracts and normalizes key game state variables:
- **Character**: Player character (normalized by max 25 characters)
- **Health**: Player health (0-166, normalized to [0,1])
- **Aggressor Bar**: Special meter (0-48, normalized to [0,1])
- **Timer**: Round timer (0-100, normalized to [0,1])

## Training Configuration

### Hyperparameters
- **Learning Rate**: 1e-4 (reduced for stability)
- **Discount Factor**: 0.99
- **Entropy Coefficient**: 0.1 (increased for exploration)
- **Value Loss Coefficient**: 0.5
- **Rollout Steps**: 128
- **Max Gradient Norm**: 0.5

### Training Process
1. Collects 128-step rollouts using current policy
2. Computes advantages using Generalized Advantage Estimation (GAE)
3. Updates actor-critic network using policy gradient and value function loss
4. Logs training metrics to TensorBoard
5. Periodically evaluates policy and saves checkpoints

## Usage

### Training
```bash
python train_ppo.py
```

### Evaluation
```bash
# Basic evaluation
python evaluate_policy.py --checkpoint checkpoints/latest_checkpoint.pth --n-episodes 20

# Deterministic evaluation with video recording
python evaluate_policy.py --checkpoint checkpoints/latest_checkpoint.pth --n-episodes 5 --deterministic --save-video ./videos

# Save results to CSV
python evaluate_policy.py --checkpoint checkpoints/latest_checkpoint.pth --n-episodes 10 --save-csv results.csv
```

### Debugging
```bash
# Analyze model weights and test environment
python debug_agent.py

# Add noise to escape local minimum
python add_noise.py
```

## Environment Configuration

- **Game**: Ultimate Mortal Kombat 3 (umk3)
- **Frame Shape**: 128x128 RGB (converted to grayscale)
- **Action Space**: Discrete (14 possible actions)
- **Step Ratio**: 4 (action repeated for 4 game frames)
- **Character**: Kitana (fixed for consistency)
- **Difficulty**: Level 2
- **Role**: Player 1

## Training Features

### Monitoring
- Real-time TensorBoard logging for losses and rewards
- Periodic policy evaluation episodes
- Model checkpointing every 100 episodes
- Training progress tracking with episode and step counters

### Stability Features
- Gradient clipping to prevent exploding gradients
- Temperature scaling for action distribution
- Proper reward normalization
- Graceful keyboard interrupt handling

## Implementation Details

The ppo implementation uses:
- Synchronous advantage actor-critic algorithm
- Combined loss function (policy loss + value loss + entropy bonus)
- Adam optimizer with gradient clipping
- Frame stacking using efficient deque data structure
- Proper environment handling with DIAMBRA Arena integration

This implementation is designed for research and experimentation with reinforcement learning in fighting game environments.

