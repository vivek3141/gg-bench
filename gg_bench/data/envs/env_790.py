import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to numbers 1-10, represented as indices 0-9
        self.action_space = spaces.Discrete(10)

        # Observation space represents the state of the game:
        # 0: Number is available in the pool
        # 1: Number is held by Player 1
        # 2: Number is held by Player 2
        self.observation_space = spaces.Box(low=0, high=2, shape=(10,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the Number Pool (all numbers available)
        self.number_pool = np.zeros(10, dtype=np.int8)

        # Each player's hand (empty at the start)
        self.player_hands = {1: [], 2: []}

        # Player 1 starts the game
        self.current_player = 1

        # Flag to indicate if the game is over
        self.done = False

        return self._get_obs(), {}  # Observation and info

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self._get_obs(), -10, True, False, {}

        # Validate the action
        if action < 0 or action >= 10:
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Check if the selected number is available in the Number Pool
        if self.number_pool[action] != 0:
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Valid action: update the game state
        self.number_pool[action] = self.current_player
        self.player_hands[self.current_player].append(action + 1)  # Numbers are 1-10

        # Check for a winning sequence
        if self._check_sequence(self.player_hands[self.current_player]):
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Check for draw (should not occur in this game)
        if np.all(self.number_pool != 0):
            self.done = True
            return self._get_obs(), 0, True, False, {}

        # Switch to the next player
        self.current_player = 3 - self.current_player  # Switches between 1 and 2

        return (
            self._get_obs(),
            0,
            False,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def render(self):
        # Create a visual representation of the game state
        output = ""
        available_numbers = [i + 1 for i in range(10) if self.number_pool[i] == 0]
        output += f"Number Pool: {available_numbers}\n"
        output += f"Player 1's Hand: {sorted(self.player_hands[1])}\n"
        output += f"Player 2's Hand: {sorted(self.player_hands[2])}\n"
        output += f"Current Player: Player {self.current_player}\n"
        return output

    def valid_moves(self):
        # Return a list of valid actions (indices of available numbers)
        return [i for i in range(10) if self.number_pool[i] == 0]

    def _get_obs(self):
        # Return a copy of the current game state as the observation
        return self.number_pool.copy()

    def _check_sequence(self, hand):
        # Check if the player's hand contains an arithmetic sequence
        if len(hand) < 3:
            return False
        hand_numbers = sorted(hand)
        n = len(hand_numbers)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = hand_numbers[i], hand_numbers[j], hand_numbers[k]
                    if b - a == c - b:
                        return True
        return False
