import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Remove Left Digit, 1 - Remove Middle Digit, 2 - Remove Right Digit
        self.action_space = spaces.Discrete(3)

        # Observations: Array of shape (6,) representing both players' digits
        # Digits range from 1 to 9; zeros represent missing digits after removal
        self.observation_space = spaces.Box(low=0, high=9, shape=(6,), dtype=np.int8)

        # Initialize state variables
        self.digits_p1 = None  # Player 1's digits
        self.digits_p2 = None  # Player 2's digits
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False  # Game over flag

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly generate unique three-digit numbers for both players
        self.digits_p1 = np.random.randint(1, 10, size=3).tolist()
        self.digits_p2 = np.random.randint(1, 10, size=3).tolist()
        self.current_player = 0  # Player 1 starts
        self.done = False

        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Get opponent's digits
        if self.current_player == 0:
            opponent_digits = self.digits_p2
        else:
            opponent_digits = self.digits_p1

        # Map action to digit index
        digit_index = self._action_to_index(len(opponent_digits), action)
        if digit_index is None:
            # Invalid action due to incorrect mapping
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Remove the selected digit and update opponent's number
        del opponent_digits[digit_index]

        # Check for win condition
        if len(opponent_digits) == 1:
            self.done = True
            reward = 1  # Current player wins
            observation = self._get_observation()
            return observation, reward, True, False, {}
        else:
            # Continue the game
            self.current_player = 1 - self.current_player  # Switch player
            reward = 0
            observation = self._get_observation()
            return observation, reward, False, False, {}

    def render(self):
        # Return a string representation of the current game state
        p1_number = "".join(str(d) for d in self.digits_p1)
        p2_number = "".join(str(d) for d in self.digits_p2)
        s = f"Player 1's number: {p1_number}\nPlayer 2's number: {p2_number}\n"
        s += f"Current turn: Player {self.current_player + 1}"
        return s

    def valid_moves(self):
        # Return a list of valid actions based on the opponent's number
        if self.current_player == 0:
            opponent_digits = self.digits_p2
        else:
            opponent_digits = self.digits_p1

        length = len(opponent_digits)
        valid_actions = []
        if length == 3:
            valid_actions = [0, 1, 2]  # All positions are valid
        elif length == 2:
            valid_actions = [0, 2]  # Only left and right positions are valid
        else:
            valid_actions = []  # No valid moves if opponent's number is a single digit
        return valid_actions

    def _get_observation(self):
        # Construct the observation array based on the players' digits
        digits_p1 = self.digits_p1 + [0] * (3 - len(self.digits_p1))
        digits_p2 = self.digits_p2 + [0] * (3 - len(self.digits_p2))
        observation = np.array(digits_p1 + digits_p2, dtype=np.int8)
        return observation

    def _action_to_index(self, number_length, action):
        # Map action to the correct digit index in the opponent's number
        if number_length == 3:
            # Actions map directly to indices
            return action
        elif number_length == 2:
            if action == 0:
                return 0  # Left digit
            elif action == 2:
                return 1  # Right digit
            else:
                return None  # Invalid action (no middle digit)
        else:
            return None  # No valid actions when number_length <= 1
