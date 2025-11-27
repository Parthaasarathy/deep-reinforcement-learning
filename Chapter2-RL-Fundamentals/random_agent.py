"""
random_agent.py
----------------------------------------
This is the MOST BASIC agent possible in Reinforcement Learning.

It does NOT learn.
It does NOT think.
It simply chooses RANDOM actions at every step.

Why is this useful?

1. It gives you a baseline: “What happens if the agent is dumb?”
2. It helps you understand the RL loop: reset → step → done.
3. It teaches you how to interact with your GridWorld environment.
4. It generates episode data you can later compare with Q-learning.

This file is written in Ultra Beginner Mode.
Every line is explained.
"""

# -----------------------------------------------------------
# 1. IMPORTS
# -----------------------------------------------------------

import gymnasium as gym
# This imports the Gymnasium library.
# Gymnasium provides:
# - env.reset()
# - env.step()
# - env.render()
# - env.action_space
# - env.observation_space

import gymnasium_env  
# This imports YOUR custom GridWorld environment.
# This line is required because importing your package triggers
# registration of "GridWorld-v0" with Gymnasium.

import numpy as np
# NumPy is the standard library for working with numbers, arrays,
# and scientific computing in Python.

import matplotlib.pyplot as plt
# Matplotlib is used to show images frame-by-frame (visualization).


# -----------------------------------------------------------
# 2. CREATE THE ENVIRONMENT
# -----------------------------------------------------------

env = gym.make("GridWorld-v0", render_mode="rgb_array")
"""
gym.make(...) loads an environment.

"GridWorld-v0" is the ID you registered in your package.

render_mode="rgb_array" means:
- The environment will return an image (RGB frame) each time
  you call env.render().

This allows us to visualize the agent moving.
"""


# -----------------------------------------------------------
# 3. RESET THE ENVIRONMENT
# -----------------------------------------------------------

obs, info = env.reset()
"""
env.reset() does 3 things:

1. Places the agent somewhere in the grid.
2. Places the target somewhere.
3. Returns the initial OBSERVATION (obs).

The observation is usually:
[agent_x, agent_y, target_x, target_y]

The second return value (info) is a dictionary with
extra information (not needed for now).
"""


# -----------------------------------------------------------
# 4. SET UP VISUALIZATION
# -----------------------------------------------------------

plt.ion()
# Turns on "interactive mode", meaning matplotlib updates frames in real-time.

fig, ax = plt.subplots()
# Creates a figure (window) and axes (area to draw in).

frame = env.render()
# Render the FIRST frame of the environment.

img = ax.imshow(frame)
# Display the first frame inside the axes.

plt.axis("off")
# Removes the axis grid, ticks, numbers – for a cleaner view.


# -----------------------------------------------------------
# 5. RUN THE EPISODE WITH RANDOM ACTIONS
# -----------------------------------------------------------

terminated = False   # True when the agent reaches the goal
truncated = False    # True if episode is cut short (timeouts)

step_count = 0       # To count how many steps this episode takes

while not (terminated or truncated):
    """
    This is THE core RL loop:

    while episode not finished:

        1. choose an action
        2. apply action → get new state and reward
        3. update visualization
        4. continue until done
    """

    # -------------------------------------------------------
    # 5A. SELECT A RANDOM ACTION
    # -------------------------------------------------------

    action = env.action_space.sample()
    """
    env.action_space.sample() returns a random valid action.

    In GridWorld:
        sample() randomly returns one of: 0, 1, 2, 3
        (right, up, left, down)
    """

    # -------------------------------------------------------
    # 5B. APPLY THE ACTION TO THE ENVIRONMENT
    # -------------------------------------------------------

    obs, reward, terminated, truncated, info = env.step(action)
    """
    env.step(action):

    Input:
        an action (0,1,2,3)

    Output:
        obs       → new observation (new positions)
        reward    → reward for this step
        terminated → True if goal reached
        truncated  → True if max steps reached
        info       → extra info (not needed)
    """

    step_count += 1

    # -------------------------------------------------------
    # 5C. UPDATE THE VISUALIZATION
    # -------------------------------------------------------

    frame = env.render()
    # Get the new rendered frame

    img.set_data(frame)
    # Update the image on screen

    fig.canvas.draw_idle()
    # Refresh the display without blocking

    plt.pause(0.15)
    # Slow down animation so humans can see movement
    # Decrease to 0.05 if you want speeding movement


# -----------------------------------------------------------
# 6. END OF EPISODE
# -----------------------------------------------------------

print("Episode finished.")
print("Steps taken:", step_count)
print("Final observation:", obs)
print("Terminated:", terminated, "Truncated:", truncated)

plt.ioff()   # Turn off interactive mode
plt.show()   # Show final frame

env.close()  # Clean up environment resources


