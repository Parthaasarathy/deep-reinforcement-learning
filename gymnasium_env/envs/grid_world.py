import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GridWorldEnv(gym.Env):
    """
    Visual GridWorld environment with optional obstacles and visited-trail rendering.

    - Square grid of size `size` x `size`.
    - Agent (A) starts at upper-left (0,0) by default.
    - Target (T) at bottom-right (size-1,size-1) by default.
    - Action space: 0=right, 1=up, 2=left, 3=down
    - Observation: Box with agent (x,y) and target (x,y) -> shape (4,)
    - Reward: -0.01 per step, +1.0 for reaching the target.
    - Episode ends when target reached or max_steps exceeded.
    - Rendering: `render(mode="rgb_array")` returns HxWx3 uint8 frame.
    - New: `visited` trail is shown; obstacles can be supplied as list of (x,y).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        size: int = 8,
        max_steps: int = 200,
        tile_pixels: int = 32,
        obstacles=None,
    ):
        super().__init__()

        assert size >= 3, "size must be >= 3"
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.tile_pixels = int(tile_pixels)

        # Action space: right, up, left, down
        self.action_space = spaces.Discrete(4)

        # Observation: agent_x, agent_y, target_x, target_y  (all ints)
        low = np.array([0, 0, 0, 0], dtype=np.int32)
        high = np.array(
            [self.size - 1, self.size - 1, self.size - 1, self.size - 1], dtype=np.int32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # State
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.target_pos = np.array([self.size - 1, self.size - 1], dtype=np.int32)
        self.steps = 0
        self.render_mode = render_mode

        # visited trail
        self.visited = set()

        # obstacles
        self.obstacles = set()
        if obstacles is not None:
            # accept list or set of (x,y)
            self.obstacles = {tuple(o) for o in obstacles}

    def _get_obs(self):
        return np.array(
            [self.agent_pos[0], self.agent_pos[1], self.target_pos[0], self.target_pos[1]],
            dtype=np.int32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # start agent at top-left, target bottom-right (you can change later)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.target_pos = np.array([self.size - 1, self.size - 1], dtype=np.int32)
        self.steps = 0

        # visited reset
        self.visited = {(int(self.agent_pos[0]), int(self.agent_pos[1]))}

        info = {}
        return self._get_obs(), info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"
        self.steps += 1

        # Map action to movement (dx, dy)
        mapping = {
            0: np.array([1, 0], dtype=np.int32),   # right
            1: np.array([0, -1], dtype=np.int32),  # up (decrease y)
            2: np.array([-1, 0], dtype=np.int32),  # left
            3: np.array([0, 1], dtype=np.int32),   # down (increase y)
        }
        move = mapping[action]
        new_pos = self.agent_pos + move

        # keep inside bounds [0, size-1]
        new_pos[0] = np.clip(new_pos[0], 0, self.size - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.size - 1)

        # prevent moving into obstacles (stay in place if target cell is blocked)
        if (int(new_pos[0]), int(new_pos[1])) in self.obstacles:
            # no movement into obstacle; agent stays in current position
            new_pos = self.agent_pos.copy()

        self.agent_pos = new_pos

        # mark visited
        self.visited.add((int(self.agent_pos[0]), int(self.agent_pos[1])))

        reached = np.array_equal(self.agent_pos, self.target_pos)
        terminated = bool(reached or (self.steps >= self.max_steps))
        truncated = False

        # reward scheme
        reward = 1.0 if reached else -0.01

        info = {}
        return self._get_obs(), reward, terminated, truncated, info

    def _render_frame(self):
        """Return an RGB image (H x W x 3) of the current grid."""
        H = self.size * self.tile_pixels
        W = self.size * self.tile_pixels
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # colors
        bg_color = np.array([240, 240, 240], dtype=np.uint8)  # light gray
        grid_color = np.array([200, 200, 200], dtype=np.uint8)
        agent_color = np.array([30, 144, 255], dtype=np.uint8)  # dodgerblue
        target_color = np.array([50, 205, 50], dtype=np.uint8)  # limegreen
        visited_color = np.array([180, 220, 255], dtype=np.uint8)  # pale blue for visited
        obstacle_color = np.array([80, 80, 80], dtype=np.uint8)  # dark gray for obstacles

        # fill background
        frame[:, :] = bg_color

        # draw grid lines (optional thin lines)
        for i in range(self.size + 1):
            x = i * self.tile_pixels
            y = i * self.tile_pixels
            # vertical line (clamp slice)
            frame[:, x : x + 1] = grid_color
            # horizontal line
            frame[y : y + 1, :] = grid_color

        # draw visited cells with a faint color (before agent/target)
        for vx, vy in self.visited:
            if 0 <= int(vx) < self.size and 0 <= int(vy) < self.size:
                x0 = int(vx) * self.tile_pixels
                y0 = int(vy) * self.tile_pixels
                frame[y0 + 2 : y0 + self.tile_pixels - 2, x0 + 2 : x0 + self.tile_pixels - 2] = visited_color

        # draw obstacles
        for ox, oy in self.obstacles:
            if 0 <= int(ox) < self.size and 0 <= int(oy) < self.size:
                x0 = int(ox) * self.tile_pixels
                y0 = int(oy) * self.tile_pixels
                frame[y0 + 1 : y0 + self.tile_pixels - 1, x0 + 1 : x0 + self.tile_pixels - 1] = obstacle_color

        # draw target square
        tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
        x0 = tx * self.tile_pixels
        y0 = ty * self.tile_pixels
        frame[y0 + 1 : y0 + self.tile_pixels - 1, x0 + 1 : x0 + self.tile_pixels - 1] = target_color

        # draw agent square on top
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        x0 = ax * self.tile_pixels
        y0 = ay * self.tile_pixels
        frame[y0 + 1 : y0 + self.tile_pixels - 1, x0 + 1 : x0 + self.tile_pixels - 1] = agent_color

        return frame

    def render(self):
        """
        Return an RGB array (H x W x 3) when render_mode == "rgb_array".
        If render_mode == "human", try to display with matplotlib if available,
        otherwise print a text fallback.
        """
        frame = self._render_frame()

        if self.render_mode == "rgb_array":
            return frame

        if self.render_mode == "human":
            try:
                import matplotlib.pyplot as plt

                plt.imshow(frame)
                plt.axis("off")
                plt.show(block=False)
                plt.pause(0.001)
            except Exception:
                # fallback: ASCII print of positions
                print(f"Agent: {self.agent_pos.tolist()}  Target: {self.target_pos.tolist()}  step: {self.steps}")
            return None

        # default behaviour: return rgb array
        return frame

    def close(self):
        # nothing to clean up; if using external viewer, close it here
        pass
