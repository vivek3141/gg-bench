import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is Discrete(9), representing selecting a number from 1 to 9
        self.action_space = spaces.Discrete(9)

        # The observation space is a Box space with values -1, 0, or 1 for each number
        # -1: Number is held by opponent
        #  0: Number is in the pool (available)
        #  1: Number is held by current player
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize observation: all numbers are in the pool (0)
        self.observation = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        return self.observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is over, no further moves are allowed
            return self.observation, 0, True, False, {}

        if action < 0 or action >= 9 or self.observation[action] != 0:
            # Invalid move: action is out of bounds or number is already taken
            self.done = True
            return (
                self.observation,
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Valid move: select the number and add it to the current player's hand
        self.observation[action] = self.current_player

        # Check for victory: can the current player form A + B = C with their numbers?
        numbers_in_hand = [
            idx + 1
            for idx, val in enumerate(self.observation)
            if val == self.current_player
        ]

        # Generate all combinations of three unique numbers from the player's hand
        victory = False
        for combo in combinations(numbers_in_hand, 3):
            a, b, c = combo
            if a + b == c or a + c == b or b + c == a:
                victory = True
                break

        if victory:
            # Current player wins
            self.done = True
            return self.observation, 1, True, False, {}
        else:
            # Switch to the other player
            self.current_player *= -1
            return self.observation, 0, False, False, {}

    def render(self):
        # Create a visual representation of the game state
        pool_numbers = [idx + 1 for idx, val in enumerate(self.observation) if val == 0]
        player1_numbers = [
            idx + 1 for idx, val in enumerate(self.observation) if val == 1
        ]
        player2_numbers = [
            idx + 1 for idx, val in enumerate(self.observation) if val == -1
        ]

        render_str = "Number Pool: " + ", ".join(map(str, pool_numbers)) + "\n"
        render_str += "Player 1 Hand: " + ", ".join(map(str, player1_numbers)) + "\n"
        render_str += "Player 2 Hand: " + ", ".join(map(str, player2_numbers)) + "\n"

        return render_str

    def valid_moves(self):
        # Return a list of indices of numbers that are still in the pool
        return [i for i in range(9) if self.observation[i] == 0]
