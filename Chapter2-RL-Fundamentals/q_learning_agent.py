"""
q_learning_agent.py
----------------------------------------
A clean and minimal Q-learning implementation for GridWorld.

You will learn:
    • how a Q-table works
    • how ε-greedy exploration works
    • how Q-values are updated
    • how an agent improves over episodes

This is the simplest and purest form of Q-learning.
"""

import gymnasium_env
import gymnasium as gym
import numpy as np
import random


# --------------------------------------------------------
#             Convert observation -> state index
# --------------------------------------------------------
def obs_to_state(obs, size):
    """
    Observation is [agent_x, agent_y, target_x, target_y]
    We convert (ax, ay, tx, ty) into a single integer state.
    """
    ax, ay, tx, ty = map(int, obs)
    return ax + ay*size + tx*(size**2) + ty*(size**3)


# --------------------------------------------------------
#                 Q-Learning Training Loop
# --------------------------------------------------------
def train_q_learning(episodes=3000, alpha=0.1, gamma=0.99, epsilon=0.2):
    env = gym.make("GridWorld-v0")
    size = env.unwrapped.size

    # total number of possible states
    num_states = size**4
    num_actions = env.action_space.n

    # Initialize Q-table with zeros
    Q = np.zeros((num_states, num_actions))

    for ep in range(episodes):
        obs, info = env.reset()
        state = obs_to_state(obs, size)

        done = False
        while not done:
            # ------------------------------
            #     ε-greedy action choice
            # ------------------------------
            if random.random() < epsilon:
                action = env.action_space.sample()  # explore
            else:
                action = np.argmax(Q[state])        # exploit

            # interact with environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = obs_to_state(next_obs, size)

            done = terminated or truncated

            # ------------------------------
            #         Q-Learning Update
            # ------------------------------
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            # move to next state
            state = next_state

        # Print every 500 episodes
        if ep % 500 == 0:
            print(f"Episode {ep}/{episodes}")

    env.close()
    return Q


# --------------------------------------------------------
#              Evaluate learned Q-table
# --------------------------------------------------------
def test_policy(Q, episodes=3):
    env = gym.make("GridWorld-v0", render_mode="human")
    size = env.unwrapped.size

    for ep in range(episodes):
        obs, info = env.reset()
        state = obs_to_state(obs, size)

        done = False
        steps = 0

        while not done:
            action = np.argmax(Q[state])   # greedy (no exploration)
            obs, reward, terminated, truncated, _ = env.step(action)
            state = obs_to_state(obs, size)
            done = terminated or truncated
            steps += 1

        print(f"Test Episode {ep+1}: reached goal in {steps} steps.")
    env.close()


# --------------------------------------------------------
#                   MAIN ENTRY POINT
# --------------------------------------------------------
if __name__ == "__main__":
    print("Training Q-learning agent...")
    Q = train_q_learning(episodes=3000)

    print("\nTesting greedy policy...")
    test_policy(Q)


