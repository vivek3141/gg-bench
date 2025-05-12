import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space corresponds to selecting a number from 1 to 30 (indices 0 to 29)
        self.action_space = spaces.Discrete(30)

        # Observation space consists of:
        # - 'number_pool': Binary indicators for numbers 1 to 30 (1 if available, 0 if used)
        # - 'top_number': The current top number on the tower (0 if tower is empty)
        self.observation_space = spaces.Dict(
            {
                "number_pool": spaces.MultiBinary(30),
                "top_number": spaces.Discrete(31),  # Values from 0 (empty tower) to 30
            }
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.number_pool = np.ones(
            30, dtype=np.int8
        )  # Numbers 1 to 30 are all available at the start
        self.top_number = 0  # Tower is empty at the start
        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = {
            "number_pool": self.number_pool.copy(),
            "top_number": self.top_number,
        }
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            # Game is already over
            return self._get_obs(), -10, True, False, {}

        # Check if the current player has legal moves
        allowable_numbers = self._get_allowable_numbers()
        if not allowable_numbers:
            # Current player cannot move, they lose
            self.done = True
            return self._get_obs(), -10, True, False, {}

        number_chosen = action + 1  # Convert action index to actual number

        # Validate the chosen number
        if self.number_pool[action] == 0 or number_chosen not in allowable_numbers:
            # Invalid move: number not available or not in allowable range
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Valid move: update state
        self.number_pool[action] = 0  # Mark the number as used
        self.top_number = number_chosen

        # Check if the opponent has any legal moves
        self.current_player = 3 - self.current_player  # Switch player
        opponent_allowable_numbers = self._get_allowable_numbers()
        if not opponent_allowable_numbers:
            # Opponent cannot move, current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Continue the game
        self.current_player = 3 - self.current_player  # Switch back to current player
        return self._get_obs(), 0, False, False, {}

    def _get_allowable_numbers(self):
        if self.top_number == 0:
            # Tower is empty, any available number can be chosen
            return [i + 1 for i in range(30) if self.number_pool[i] == 1]
        else:
            min_allowed = self.top_number + 1
            max_allowed = min(self.top_number + 5, 30)
            return [
                num
                for num in range(min_allowed, max_allowed + 1)
                if self.number_pool[num - 1] == 1
            ]

    def _get_obs(self):
        return {"number_pool": self.number_pool.copy(), "top_number": self.top_number}

    def render(self):
        available_numbers = [str(i + 1) for i in range(30) if self.number_pool[i] == 1]
        number_pool_str = "Available Numbers: " + ", ".join(available_numbers)
        tower_str = "Tower: " + (
            "Empty" if self.top_number == 0 else f"Top number is {self.top_number}"
        )
        current_player_str = f"Current Player: Player {self.current_player}"
        return f"{number_pool_str}\n{tower_str}\n{current_player_str}"

    def valid_moves(self):
        allowable_numbers = self._get_allowable_numbers()
        # Convert allowable numbers to action indices (subtract 1)
        return [num - 1 for num in allowable_numbers]
