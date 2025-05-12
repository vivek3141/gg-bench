import numpy as np
import gymnasium as gym
from gymnasium import spaces
import itertools


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space
        # Actions 0-119: Trapper's trap placements (120 combinations)
        # Actions 120-122: Runner's moves (move forward 1, 2, or 3 steps)
        self.action_space = spaces.Discrete(123)

        # Define observation space
        # obs[0]: current_player (0 for Trapper, 1 for Runner)
        # obs[1]: Runner's current position (0 to 10)
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.int32)

        # Store all possible combinations of traps for the Trapper
        self.trap_combinations = list(itertools.combinations(range(1, 11), 3))

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 0  # 0 for Trapper, 1 for Runner
        self.runner_position = 0  # Runner starts at cell 0
        self.traps = []  # Traps are empty until Trapper places them
        self.done = False
        self.info = {}
        observation = np.array(
            [self.current_player, self.runner_position], dtype=np.int32
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        reward = -10  # Default reward for a valid move
        terminated = False
        truncated = False

        if self.done:
            # If the game is already over
            return (
                np.array([self.current_player, self.runner_position], dtype=np.int32),
                reward,
                terminated,
                truncated,
                self.info,
            )

        if self.current_player == 0:
            # Trapper's turn
            if action < 0 or action >= 120:
                # Invalid action
                reward = -10
                self.done = True
                self.info["winner"] = "Runner"
                return (
                    np.array(
                        [self.current_player, self.runner_position], dtype=np.int32
                    ),
                    reward,
                    True,
                    False,
                    self.info,
                )
            # Set traps based on action index
            self.traps = self.trap_combinations[action]
            self.current_player = 1  # Switch to Runner
            observation = np.array(
                [self.current_player, self.runner_position], dtype=np.int32
            )
            return observation, reward, terminated, truncated, self.info

        elif self.current_player == 1:
            # Runner's turn
            if action < 120 or action > 122:
                # Invalid action
                reward = -10
                self.done = True
                self.info["winner"] = "Trapper"
                return (
                    np.array(
                        [self.current_player, self.runner_position], dtype=np.int32
                    ),
                    reward,
                    True,
                    False,
                    self.info,
                )
            # Map action to move steps
            move_steps = action - 119  # action 120->1, 121->2, 122->3
            # Adjust move if it would go beyond cell 10
            if self.runner_position + move_steps > 10:
                move_steps = 10 - self.runner_position
            self.runner_position += move_steps

            if self.runner_position in self.traps:
                # Runner steps on a trap
                reward = -10  # Reward for Runner
                self.done = True
                self.info["winner"] = "Trapper"
                observation = np.array(
                    [self.current_player, self.runner_position], dtype=np.int32
                )
                return observation, reward, True, False, self.info

            elif self.runner_position == 10:
                # Runner reaches the finish line
                reward = 1  # Runner wins
                self.done = True
                self.info["winner"] = "Runner"
                observation = np.array(
                    [self.current_player, self.runner_position], dtype=np.int32
                )
                return observation, reward, True, False, self.info

            else:
                # Game continues
                observation = np.array(
                    [self.current_player, self.runner_position], dtype=np.int32
                )
                reward = -10
                return observation, reward, terminated, truncated, self.info

    def render(self):
        path = ""
        for i in range(1, 11):
            if i == self.runner_position:
                path += "[R]"
            else:
                path += "[ ]"
        return f"Current player: {'Trapper' if self.current_player == 0 else 'Runner'}\nPath: {path}\n"

    def valid_moves(self):
        if self.current_player == 0:
            # Trapper's valid moves are actions 0-119
            return list(range(0, 120))
        elif self.current_player == 1:
            # Runner's valid moves are actions 120-122
            # Adjust for possible overstepping cell 10
            max_move = min(3, 10 - self.runner_position)
            return [119 + i for i in range(1, max_move + 1)]
        else:
            return []
