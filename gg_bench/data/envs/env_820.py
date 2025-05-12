import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(10)  # Numbers 1 to 10, indices 0 to 9

        # Observation space consists of:
        # - Indices 0-9: Numbers 1-10 availability and selection status
        #     - 0: Available
        #     - 1: Selected by Player 1
        #     - -1: Selected by Player 2
        # - Index 10: Cumulative total of Player 1
        # - Index 11: Cumulative total of Player 2
        self.observation_space = spaces.Box(
            low=-1, high=30, shape=(12,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # All numbers from 1 to 10 are initially available (0)
        self.numbers = np.zeros(10, dtype=np.int32)

        # Cumulative totals for players: [Player 1 total, Player 2 total]
        self.totals = np.array([0, 0], dtype=np.int32)

        # Player 1 starts first (1 for Player 1, -1 for Player 2)
        self.current_player = 1

        # Game is not done at the start
        self.done = False

        observation = self._get_obs()
        info = {}
        return observation, info  # Return observation and info

    def step(self, action):
        info = {}
        # Check if action is valid
        if self.done or action not in range(10) or self.numbers[action] != 0:
            self.done = True
            reward = -10  # Penalty for invalid move
            observation = self._get_obs()
            return (
                observation,
                reward,
                self.done,
                False,
                info,
            )  # Return terminated=False since not truncated

        # Valid action; update game state
        number_selected = action + 1  # Number corresponds to action index + 1
        self.numbers[action] = (
            self.current_player
        )  # Mark number as selected by current player

        # Update the cumulative total for the current player
        player_index = 0 if self.current_player == 1 else 1
        self.totals[player_index] += number_selected

        # Check for winning condition
        current_total = self.totals[player_index]
        reward = 0  # Default reward

        if current_total > 15 and current_total % 2 == 0:
            # Current player wins
            self.done = True
            reward = 1
        elif np.all(self.numbers != 0):
            # All numbers have been selected, check for highest total
            other_player_index = 1 - player_index
            other_total = self.totals[other_player_index]
            if current_total > other_total:
                # Current player wins
                self.done = True
                reward = 1
            elif current_total == other_total:
                # Tie, last player to take a turn loses
                self.done = True
                reward = -1  # Current player loses
            else:
                # Current player loses
                self.done = True
                reward = -1
        else:
            # Game continues; switch player
            self.current_player *= -1

        observation = self._get_obs()
        return (
            observation,
            reward,
            self.done,
            False,
            info,
        )  # Return observation, reward, terminated, truncated, info

    def render(self):
        lines = []
        lines.append("Available Numbers:")
        for i in range(10):
            status = self.numbers[i]
            if status == 0:
                lines.append(f"{i+1} ", end="")
        lines.append(f"\nPlayer 1 Total: {self.totals[0]}")
        lines.append(f"Player 2 Total: {self.totals[1]}")
        lines.append(
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"
        )
        return "\n".join(lines)

    def valid_moves(self):
        return [i for i in range(10) if self.numbers[i] == 0]

    def _get_obs(self):
        # Observation is an array combining numbers status and cumulative totals
        observation = np.concatenate([self.numbers, self.totals])
        return observation
