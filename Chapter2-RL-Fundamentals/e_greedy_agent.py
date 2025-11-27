"""
e_greedy_agent.py
----------------------------------------
A simple ε-greedy agent for GridWorld.

This agent does NOT learn Q-values yet.
Instead:
    - With probability ε => it explores (random action)
    - With probability 1 - ε => it exploits (moves greedily toward the target)

This agent is the conceptual bridge between:
    random agent  ->  ε-greedy agent  ->  Q-learning agent
"""

import gymnasium_env
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


# ------------------------------
# Helper: greedy move calculation
# ------------------------------
def greedy_action(agent, target):
    ax, ay = agent
    tx, ty = target

    # Move horizontally first toward target
    if tx > ax: return 0     # right
    if tx < ax: return 2     # left

    # Then vertical movement
    if ty > ay: return 3     # down
    if ty < ay: return 1     # up

    return 0  # already on target (should not happen)


# ----------------------------------------------------
#                MAIN AGENT LOOP
# ----------------------------------------------------
def run_e_greedy_agent(epsilon=0.3):
    env = gym.make("GridWorld-v0", render_mode="rgb_array")
    obs, info = env.reset()

    plt.ion()
    fig, ax = plt.subplots()
    img = ax.imshow(env.render())
    plt.axis("off")

    terminated = False
    truncated = False

    steps = 0

    while not (terminated or truncated):
        agent_pos = (int(obs[0]), int(obs[1]))
        target_pos = (int(obs[2]), int(obs[3]))

        # ----------------------------
        #   ε-GREEDY ACTION CHOICE
        # ----------------------------
        if np.random.random() < epsilon:
            action = env.action_space.sample()   # explore
        else:
            action = greedy_action(agent_pos, target_pos)  # exploit

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        # Render frame
        frame = env.render()
        img.set_data(frame)
        fig.canvas.draw_idle()
        plt.pause(0.15)

    print(f"Episode finished in {steps} steps.")
    plt.ioff()
    env.close()


if __name__ == "__main__":
    run_e_greedy_agent(epsilon=0.3)

