import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Numbers from 2 to 50 correspond to actions 0 to 48
        self.action_space = spaces.Discrete(49)
        # Observation includes available numbers and last number played (normalized)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(50,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(49, dtype=np.float32)  # Numbers 2 to 50
        self.chain = []
        self.last_number = None
        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map action to actual number
        number = action + 2

        # Check if number is available
        if action < 0 or action >= 49 or self.available_numbers[action] == 0:
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Check if the move is valid
        if self.last_number is not None:
            if self.last_number % number != 0 and number % self.last_number != 0:
                reward = -10
                self.done = True
                return self._get_observation(), reward, True, False, {}

        # Valid move, update the state
        self.available_numbers[action] = 0
        self.chain.append(number)
        self.last_number = number

        # Check if the next player has any valid moves
        self.current_player *= -1  # Switch player
        if not self._has_valid_moves():
            reward = 1  # Current player wins
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Continue the game
        reward = 0
        return self._get_observation(), reward, False, False, {}

    def render(self):
        chain_str = " → ".join(map(str, self.chain)) if self.chain else "Empty"
        available_numbers = [i + 2 for i in range(49) if self.available_numbers[i] == 1]
        render_str = f"Current Chain: {chain_str}\n"
        render_str += f"Last Number Played: {self.last_number}\n"
        render_str += f"Available Numbers: {available_numbers}\n"
        return render_str

    def valid_moves(self):
        valid_actions = []
        for action in range(49):
            if self.available_numbers[action] == 1:
                number = action + 2
                if self.last_number is None:
                    valid_actions.append(action)
                elif self.last_number % number == 0 or number % self.last_number == 0:
                    valid_actions.append(action)
        return valid_actions

    def _has_valid_moves(self):
        for action in range(49):
            if self.available_numbers[action] == 1:
                number = action + 2
                if self.last_number is None:
                    return True
                if self.last_number % number == 0 or number % self.last_number == 0:
                    return True
        return False

    def _get_observation(self):
        obs = np.zeros(50, dtype=np.float32)
        obs[:49] = self.available_numbers
        obs[49] = (self.last_number or 0) / 50  # Normalize last number
        return obs
