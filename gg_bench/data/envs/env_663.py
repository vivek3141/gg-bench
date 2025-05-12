import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: choices from 1 to 9 inclusive
        self.action_space = spaces.Discrete(
            9
        )  # actions are 0 to 8, corresponding to choices 1 to 9

        # Define observation space: Life Numbers of both players
        # Each life number ranges from 0 to 100
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([100, 100]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_life = [100, 100]  # Player 1 and Player 2 Life Numbers
        self.current_player = 0  # Index: 0 for Player 1, 1 for Player 2
        self.done = False
        return np.array(self._get_obs()), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return np.array(self._get_obs()), -10, True, False, {}

        # Convert action to choice number (1 to 9)
        choice = action + 1

        # Get indices for current player and opponent
        current = self.current_player
        opponent = 1 - self.current_player

        # Check divisibility
        if self.player_life[opponent] % choice == 0:
            # Subtract choice from opponent's Life Number
            self.player_life[opponent] -= choice
            # Ensure Life Number doesn't go below zero
            if self.player_life[opponent] < 0:
                self.player_life[opponent] = 0

            # Check for win condition
            if self.player_life[opponent] == 0:
                self.done = True
                reward = 1  # Current player wins
                return np.array(self._get_obs()), reward, True, False, {}
            else:
                reward = -10  # Valid move
        else:
            # Subtract choice from current player's Life Number
            self.player_life[current] -= choice
            # Ensure Life Number doesn't go below zero
            if self.player_life[current] < 0:
                self.player_life[current] = 0

            # Check if current player reduced their own Life Number to zero
            if self.player_life[current] == 0:
                self.done = True
                reward = -10  # Current player loses
                return np.array(self._get_obs()), reward, True, False, {}
            else:
                reward = -10  # Valid move

        # Switch to the other player
        self.current_player = opponent

        return np.array(self._get_obs()), reward, False, False, {}

    def render(self):
        board_str = f"Player 1 Life Number: {self.player_life[0]}\n"
        board_str += f"Player 2 Life Number: {self.player_life[1]}\n"
        board_str += f"Player {self.current_player + 1}'s turn.\n"
        return board_str

    def valid_moves(self):
        # Return list of valid actions (0 to 8 corresponding to choices 1 to 9)
        return list(range(9))

    def _get_obs(self):
        # Observation is [current player's life, opponent's life]
        current = self.current_player
        opponent = 1 - self.current_player
        return [self.player_life[current], self.player_life[opponent]]
