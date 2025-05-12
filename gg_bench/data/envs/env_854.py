import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(9), actions 0-8 correspond to selecting numbers 1-9
        self.action_space = spaces.Discrete(9)

        # Observation space: Box(low=0, high=50, shape=(10,), dtype=np.float32)
        # Elements 0-8: availability of numbers 1-9 (1 if available, 0 if not)
        # Element 9: current total sum (0 to 50)
        self.observation_space = spaces.Box(
            low=0, high=50, shape=(10,), dtype=np.float32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = [1] * 9  # 1 if number i+1 is available
        self.total_sum = 0
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = self.get_observation()
        info = {}
        return observation, info  # Return observation and info

    def get_observation(self):
        # Return the observation array
        obs = np.array(self.available_numbers + [self.total_sum], dtype=np.float32)
        return obs

    def step(self, action):
        if self.done:
            # Game is already over
            return self.get_observation(), -10, True, False, {}

        number = action + 1  # Map action to number (0-8) -> (1-9)

        if action < 0 or action >= 9 or self.available_numbers[action] == 0:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            return self.get_observation(), reward, True, False, {}

        # Valid move
        self.available_numbers[action] = 0  # Mark number as used
        self.total_sum += number

        if self.total_sum == 50:
            # Current player wins by reaching exactly 50
            self.done = True
            reward = 1
            return self.get_observation(), reward, True, False, {}
        elif self.total_sum > 50:
            # Current player loses by exceeding 50
            self.done = True
            reward = -10
            return self.get_observation(), reward, True, False, {}
        elif sum(self.available_numbers) == 0:
            # All numbers have been used
            self.done = True
            # Last player to move wins
            reward = 1
            return self.get_observation(), reward, True, False, {}
        else:
            # Game continues
            # Switch to next player
            self.current_player = 3 - self.current_player  # Switch between 1 and 2
            reward = 0
            return self.get_observation(), reward, False, False, {}

    def render(self):
        s = f"Total Sum: {self.total_sum}\n"
        s += "Available Numbers: "
        available_nums = [
            str(i + 1) for i in range(9) if self.available_numbers[i] == 1
        ]
        s += " ".join(available_nums) + "\n"
        s += f"Player {self.current_player}'s turn."
        return s

    def valid_moves(self):
        # Return a list of valid actions (indices of available numbers)
        return [i for i in range(9) if self.available_numbers[i] == 1]
