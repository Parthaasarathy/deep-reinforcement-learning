"""
Deep Q-Network (DQN) Agent for GridWorld
----------------------------------------

This script trains a Deep Q-Network (neural network) to solve your GridWorld
environment.  
It uses:
    - experience replay
    - epsilon-greedy exploration
    - target networks
    - PyTorch neural networks
    - batch training

Everything is written in beginner-friendly style.
"""

import random
import numpy as np
import gymnasium as gym
import gymnasium_env  # IMPORTANT: ensures GridWorld-v0 is registered

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# ---------------------------------------------------------------------------
# 1. Simple Neural Network to Approximate Q-Values
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# 2. Replay Buffer (stores experiences)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store (s,a,r,s',done) tuple."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 3. DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=50
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        # Main network & Target network
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()

        self.train_step = 0

    # -------------------------------------------------------------
    # Convert observation to tensor
    # -------------------------------------------------------------
    def preprocess(self, obs):
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    # -------------------------------------------------------------
    # Action selection (epsilon-greedy)
    # -------------------------------------------------------------
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = self.preprocess(state)
        q_values = self.policy_net(state_t)
        return torch.argmax(q_values).item()

    # -------------------------------------------------------------
    # Update / Learning step
    # -------------------------------------------------------------
    def train(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Q(s,a)
        q_values = self.policy_net(states).gather(1, actions)

        # max_a' Q_target(s',a')
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)

        # target: r + γ * max Q(s',a')(1-done)
        target = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ---------------------------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------------------------

def train_dqn(episodes=300):
    env = gym.make("GridWorld-v0")
    state_dim = 4     # (agent_x, agent_y, target_x, target_y)
    action_dim = 4    # up, down, left, right

    agent = DQNAgent(state_dim, action_dim)

    rewards_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0

        for step in range(200):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            ep_reward += reward

            if done:
                break

        rewards_history.append(ep_reward)

        print(f"Episode {ep+1}/{episodes} | Reward: {ep_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    # Save model
    torch.save(agent.policy_net.state_dict(), "Chapter3-DQN/models/dqn_model.pt")
    print("Training complete. Model saved to Chapter3-DQN/models/dqn_model.pt")

    return rewards_history


# ---------------------------------------------------------------------------
# 5. Execute training when running this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_dqn(episodes=300)

