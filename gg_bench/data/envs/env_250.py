import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 to 5 corresponding to:
        # 0: Left, remove 1
        # 1: Left, remove 2
        # 2: Left, remove 3
        # 3: Right, remove 1
        # 4: Right, remove 2
        # 5: Right, remove 3
        self.action_space = spaces.Discrete(6)

        # Observation: [left_index, right_index]
        self.observation_space = spaces.Box(low=1, high=10, shape=(2,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.left_index = 1  # Left end number
        self.right_index = 10  # Right end number
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = np.array([self.left_index, self.right_index], dtype=np.int32)
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is over
            observation = np.array([self.left_index, self.right_index], dtype=np.int32)
            return observation, -10, True, False, {}

        # Map action to side and quantity
        if action < 0 or action >= 6:
            # Invalid action index
            self.done = True
            observation = np.array([self.left_index, self.right_index], dtype=np.int32)
            return observation, -10, True, False, {}

        side = "left" if action < 3 else "right"
        quantity = (action % 3) + 1

        # Check if move is valid
        remaining_numbers = self.right_index - self.left_index + 1

        # Check if quantity is valid given remaining numbers
        max_quantity = min(3, remaining_numbers)
        if quantity > max_quantity:
            # Invalid move, cannot remove more numbers than available
            self.done = True
            observation = np.array([self.left_index, self.right_index], dtype=np.int32)
            return observation, -10, True, False, {}

        # Apply the action
        if side == "left":
            self.left_index += quantity
        else:
            self.right_index -= quantity

        # After move, check remaining numbers
        remaining_numbers = self.right_index - self.left_index + 1

        observation = np.array([self.left_index, self.right_index], dtype=np.int32)

        if remaining_numbers == 0:
            # The last number was removed, current player loses
            self.done = True
            reward = -10
            terminated = True
            truncated = False
            return observation, reward, terminated, truncated, {}
        elif remaining_numbers == 1:
            # Current player wins, as opponent will have to remove the last number
            self.done = True
            reward = 1
            terminated = True
            truncated = False
            return observation, reward, terminated, truncated, {}
        else:
            # Game continues
            self.current_player *= -1  # Switch player
            reward = -10  # Negative reward for valid move
            terminated = False
            truncated = False
            return observation, reward, terminated, truncated, {}

    def render(self):
        # Return a visual representation of the environment state as a string
        number_line = list(range(self.left_index, self.right_index + 1))
        number_line_str = "[" + " ".join(map(str, number_line)) + "]"
        return number_line_str

    def valid_moves(self):
        # Return a list of valid moves as indices of the action_space
        remaining_numbers = self.right_index - self.left_index + 1
        valid_actions = []
        max_quantity = min(3, remaining_numbers)
        for action in range(6):
            side = "left" if action < 3 else "right"
            quantity = (action % 3) + 1
            if quantity > max_quantity:
                continue  # Cannot remove more numbers than available
            valid_actions.append(action)
        return valid_actions
