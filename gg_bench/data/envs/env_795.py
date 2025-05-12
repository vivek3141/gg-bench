import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(16)

        self.observation_space = spaces.Box(
            low=-1.0, high=50.0, shape=(18,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the grid: 0 means available, 1 means taken by agent, -1 taken by opponent
        self.grid = np.zeros(16, dtype=np.int8)
        self.agent_sum = 0
        self.opponent_sum = 0
        self.done = False
        # Set initial observation
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}

        if action < 0 or action >= 16:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        if self.grid[action] != 0:
            # Invalid move, number already taken
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Agent's move
        number = action + 1  # Numbers from 1 to 16
        self.grid[action] = 1  # Mark number as taken by agent
        self.agent_sum += number

        if self.agent_sum >= 34:
            # Agent loses
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check if there are any valid moves left for opponent
        if not self._has_valid_moves():
            # No valid moves left, game ends in a draw (but per game rules, there are no draws)
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Opponent's move
        opponent_action = self._opponent_policy()
        if opponent_action is not None:
            opponent_number = opponent_action + 1
            self.grid[opponent_action] = -1  # Mark number as taken by opponent
            self.opponent_sum += opponent_number

            if self.opponent_sum >= 34:
                # Agent wins
                self.done = True
                return self._get_observation(), 1, True, False, {}
        else:
            # No valid moves left for opponent, game continues
            pass

        # Per the rules, each valid move gives a reward of -10
        return self._get_observation(), -10, False, False, {}

    def render(self):
        grid_visual = ""
        for i in range(4):
            grid_visual += "+----+----+----+----+\n"
            for j in range(4):
                idx = i * 4 + j
                cell_value = self.grid[idx]
                if cell_value == 0:
                    cell_content = f"{idx+1:2d}"
                elif cell_value == 1:
                    cell_content = " A "
                elif cell_value == -1:
                    cell_content = " O "
                else:
                    cell_content = "   "
                grid_visual += f"| {cell_content} "
            grid_visual += "|\n"
        grid_visual += "+----+----+----+----+\n"
        grid_visual += f"Agent Sum: {self.agent_sum}\n"
        grid_visual += f"Opponent Sum: {self.opponent_sum}\n"
        return grid_visual

    def valid_moves(self):
        return [i for i in range(16) if self.grid[i] == 0]

    def _get_observation(self):
        # Create an observation array
        observation = np.zeros(18, dtype=np.float32)
        # Grid state
        observation[0:16] = self.grid
        # Agent's sum
        observation[16] = self.agent_sum
        # Opponent's sum
        observation[17] = self.opponent_sum
        return observation

    def _has_valid_moves(self):
        return np.any(self.grid == 0)

    def _opponent_policy(self):
        # Simple opponent policy: randomly choose a valid move
        valid_moves = self.valid_moves()
        if not valid_moves:
            return None
        opponent_action = self.np_random.choice(valid_moves)
        return opponent_action
