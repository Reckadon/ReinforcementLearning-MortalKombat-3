# DQN Agent for Ultimate Mortal Kombat 3

This branch contains a Deep Q-Network (DQN) reinforcement learning implementation for playing **Ultimate Mortal Kombat 3** using the **DIAMBRA Arena** environment. It is adapted with minimal changes from an existing A2C implementation so you can directly compare algorithmic effects in the same environment and pipeline.

---

## Architecture Overview

### Neural Network Architecture

* **Dual-Input Design**: Combines visual frame input (stacked grayscale frames) with compact game-state features.
* **CNN Encoder**: Same 3-layer convolutional encoder used in the A2C code to process stacked frames.
* **MLP Encoder**: Small MLP to encode normalized game-state features (character, health, aggressor bar, timer).
* **Q-Head**: Single head that outputs Q-values for each discrete action in UMK3.

### Key Features

* Reuses the same preprocessing pipeline (grayscale conversion, normalization, 4-frame stacking)
* Replay buffer for off-policy minibatch updates
* Epsilon-greedy action selection with linear decay schedule
* Target network for stabilized Q-learning (periodic hard updates)
* Checkpointing & TensorBoard logging

---

## DQN Algorithm Description

### Overview

Deep Q-Network (DQN) is a value-based off-policy RL algorithm. It approximates the action-value function Q(s, a) with a neural network and learns by minimizing the temporal-difference (TD) error:

```
L(θ) = E[(r + γ * max_{a'} Q_target(s', a') - Q_online(s, a))^2]
```

Key practical stabilizers used in this implementation:

* **Replay Buffer** stores transitions and breaks correlations between sequential samples by sampling uniformly.
* **Target Network** is a delayed copy of the online network used to compute stable targets.
* **Epsilon-Greedy Exploration** starts with high exploration (ε≈1) and decays to a small ε (≈0.01).

### Core Components

#### Q-Network (Online)

* Inputs: stacked frames + normalized features
* Output: vector of Q-values (one per discrete action)
* Loss: MSE between predicted Q(s,a) and TD target

#### Target Network

* A frozen copy of the online network that is periodically updated (hard update) to reduce target oscillation

#### Replay Buffer

* Stores transitions `(state, action, reward, next_state, done)`
* Samples random minibatches for training

#### Epsilon-Greedy Policy

* With probability ε choose a random action; otherwise choose `argmax_a Q(s,a)`
* ε decays linearly from `epsilon_start` to `epsilon_final` over `epsilon_decay` steps

---

## Architecture Diagram

![diagram](/archi_dqn.png)

## Input Processing

### Frame Processing

* Converts input RGB frames (e.g. 128×128×3) to grayscale using luminance weights
* Normalizes pixel values to [0,1]
* Stacks 4 consecutive frames into a tensor of shape `(4, H, W)`

### Feature Processing

Extracts and normalizes game-state variables to the range [0,1]:

* **Character**: normalized by 25 (number of characters)
* **Health**: normalized by 166
* **Aggressor Bar**: normalized by 48
* **Timer**: normalized by 100

These are concatenated with the CNN embedding before the Q-head.

---

## Hyperparameters (defaults in the code)

* `stack_size`: 4
* `learning_rate`: 1e-4
* `gamma`: 0.99
* `replay_capacity`: 100000
* `batch_size`: 64
* `train_freq`: 4 (train every 4 environment steps)
* `target_update`: 1000 (hard update frequency in steps)
* `epsilon_start`: 1.0
* `epsilon_final`: 0.01
* `epsilon_decay`: 100000 (steps)
* `max_grad_norm`: 0.5
* Logging: TensorBoard under `runs/dqn_umk3`

These were chosen to be conservative and stable for this environment; tune them empirically for faster convergence.

---

## Training Procedure

1. Initialize online and target networks (target ← online)
2. Reset environment and frame stack
3. Repeat until `max_episodes`:

   * Select action with epsilon-greedy policy
   * Step environment and collect `(s, a, r, s', done)`
   * Store transition in replay buffer (store tensors on CPU to save GPU memory)
   * Every `train_freq` steps, sample a minibatch and perform a gradient step to minimize TD MSE loss
   * Every `target_update` steps, copy online weights to target network
   * Periodically evaluate (greedy, ε=0) and save checkpoints

---

## Usage

### Train

```bash
diambra run python train_dqn.py
```

### Evaluate (greedy policy)

### TensorBoard

```bash
tensorboard --logdir runs
```

---

## Visualization

![rewards](/dqn_reward.png)
