import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space consists of 8 possible actions, corresponding to the numbers 2 to 9
        self.action_space = spaces.Discrete(8)  # Actions 0-7 correspond to numbers 2-9

        # The observation space is a binary array indicating the presence (1) or absence (0) of each number
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the shared list with numbers 2 to 9 present
        self.shared_list = np.ones(8, dtype=np.float32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.shared_list.copy(), {}  # Return observation and info dict

    def step(self, action):
        if self.done:
            # If the game is already over, return with a penalty
            return self.shared_list.copy(), -10, True, False, {}
        if action < 0 or action >= 8:
            # Invalid action outside the action space
            self.done = True
            return self.shared_list.copy(), -10, True, False, {}
        if self.shared_list[action] == 0:
            # Invalid move: chosen number is not in the shared list
            self.done = True
            return self.shared_list.copy(), -10, True, False, {}

        # Valid move
        chosen_number = action + 2  # Map action to the actual number (2 to 9)

        # Remove the chosen number and all numbers divisible by it
        self.shared_list[action] = 0  # Remove chosen number
        for i in range(8):
            number = i + 2
            if self.shared_list[i] == 1 and number % chosen_number == 0:
                self.shared_list[i] = 0  # Remove divisible number

        # Check if the game has ended
        if np.all(self.shared_list == 0):
            # Current player wins
            self.done = True
            return self.shared_list.copy(), 1, True, False, {}
        else:
            # Game continues
            self.current_player *= -1  # Switch player
            return self.shared_list.copy(), -10, False, False, {}

    def render(self):
        # Create a visual representation of the shared list
        numbers = [str(i + 2) if self.shared_list[i] == 1 else "X" for i in range(8)]
        return "Shared List: [" + ", ".join(numbers) + "]"

    def valid_moves(self):
        # Return a list of valid action indices corresponding to available numbers
        return [i for i in range(8) if self.shared_list[i] == 1]
