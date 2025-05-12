import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(29)  # Actions correspond to numbers 2 to 30

        # Observation space:
        # Index 0: Current number (scaled between 2 and 30)
        # Index 1: Current player (-1 or 1)
        # Indices 2-30: Indicators for numbers 2 to 30 (1 if available, 0 if not)
        self.observation_space = spaces.Box(
            low=np.array([2, -1] + [0] * 29, dtype=np.float32),
            high=np.array([30, 1] + [1] * 29, dtype=np.float32),
            dtype=np.float32,
        )

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.number_pool = list(range(2, 31))  # Numbers from 2 to 30 inclusive
        self.current_player = 1  # Player 1 starts after starting number is selected

        # Randomly select starting number from the number pool
        self.current_number = self.np_random.choice(self.number_pool)
        self.number_pool.remove(self.current_number)

        # Initialize number pool indicators
        self.number_pool_indicator = np.zeros(29, dtype=np.float32)
        for num in self.number_pool:
            idx = num - 2  # Map number to index 0-28
            self.number_pool_indicator[idx] = 1.0

        self.done = False
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        selected_number = action + 2  # Map action index to number (2 to 30)

        # Check if selected number is available in the number pool
        if selected_number not in self.number_pool:
            # Invalid move: number not in pool
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Check if move is valid according to game rules
        if not self._is_valid_move(selected_number):
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Perform the move
        self.number_pool.remove(selected_number)
        idx = selected_number - 2
        self.number_pool_indicator[idx] = 0.0
        self.current_number = selected_number

        # Switch to the next player
        self.current_player *= -1

        # Check if the next player has valid moves
        if not self._has_valid_moves():
            self.done = True
            reward = 1  # Current player wins
        else:
            reward = 0

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        # Return a string representation of the current game state
        render_str = "--- Factor Frenzy ---\n"
        render_str += f"Current Number: {self.current_number}\n"
        render_str += f"Player {1 if self.current_player == 1 else 2}'s turn.\n"
        render_str += f"Available Numbers: {sorted(self.number_pool)}\n"
        return render_str

    def valid_moves(self):
        # Return a list of valid moves as action indices
        valid_moves = []
        for num in self.number_pool:
            if self._is_valid_move(num):
                action_idx = num - 2
                valid_moves.append(action_idx)
        return valid_moves

    def _is_valid_move(self, selected_number):
        # Check if selected number is a factor of the current number (excluding 1 and itself)
        if selected_number != 1 and selected_number != self.current_number:
            if self.current_number % selected_number == 0:
                return True

        # Check if selected number is a multiple of the current number
        if selected_number % self.current_number == 0:
            return True

        return False

    def _has_valid_moves(self):
        # Check if there is at least one valid move available
        for num in self.number_pool:
            if self._is_valid_move(num):
                return True
        return False

    def _get_obs(self):
        # Construct the observation array
        observation = np.concatenate(
            ([self.current_number, self.current_player], self.number_pool_indicator)
        )
        return observation
