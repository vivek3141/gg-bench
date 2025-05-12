import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space allows the agent to pick a digit to remove from positions 0 to 3
        self.action_space = spaces.Discrete(4)

        # The observation is the Shared Number represented as four digits
        # The digits are stored in an array of length 4, padded with zeros if necessary
        self.observation_space = spaces.Box(low=0, high=9, shape=(4,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Generate a random four-digit number between 1000 and 9999
        self.shared_number = np.random.randint(1000, 10000)
        # Convert the number to a list of digits
        self.digits = [int(d) for d in str(self.shared_number)]
        self.current_player = 1  # Player 1 starts
        self.done = False
        info = {}
        return self.get_observation(), info  # Return observation and info

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action: current player loses
            reward = -10
            self.done = True
            return self.get_observation(), reward, True, False, {}

        # Apply the action
        new_digits = self.digits[:action] + self.digits[action + 1 :]
        self.digits = new_digits

        if len(self.digits) == 1:
            # Current player wins by reducing to a single-digit number
            reward = 1
            self.done = True
            return self.get_observation(), reward, True, False, {}
        else:
            # Switch to next player
            self.current_player = 3 - self.current_player  # Switch between 1 and 2

            # Check if the next player has any valid moves
            if not self.valid_moves():
                # Next player cannot make a valid move; current player wins
                reward = 1
                self.done = True
                return self.get_observation(), reward, True, False, {}
            else:
                # Game continues
                reward = 0
                return self.get_observation(), reward, False, False, {}

    def render(self):
        shared_number_str = "".join(str(d) for d in self.digits)
        print(f"Current Shared Number: {shared_number_str}")
        print(f"Current Player: Player {self.current_player}")

    def valid_moves(self):
        valid_actions = []
        for idx in range(len(self.digits)):
            new_digits = self.digits[:idx] + self.digits[idx + 1 :]
            if len(new_digits) == 0:
                continue  # Cannot remove the last digit to leave no number
            if new_digits[0] == 0:
                continue  # Resulting number would start with zero; invalid move
            # Valid move
            valid_actions.append(idx)
        return valid_actions

    def get_observation(self):
        obs = np.array(self.digits + [0] * (4 - len(self.digits)), dtype=np.int32)
        return obs
