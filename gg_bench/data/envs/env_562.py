import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the action space
        # Actions from 0 to 17
        # 0-8: Remove (action + 1) digits from the start
        # 9-17: Remove (action - 9 + 1) digits from the end
        self.max_digits = 9  # Maximum number of digits in N
        self.action_space = spaces.Discrete(18)

        # Define the observation space
        # Observation is a vector of length 9, digits 1-9 or -1 for padding
        self.observation_space = spaces.Box(low=-1, high=9, shape=(9,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = "987654321"
        self.state = [int(d) for d in self.N]  # Convert N to a list of digits
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_observation(), {}

    def _get_observation(self):
        # Pad the state with -1 to a fixed size of 9 digits
        padded_state = self.state + [-1] * (self.max_digits - len(self.state))
        return np.array(padded_state, dtype=np.int32)

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if N is empty at the start of the turn
        if len(self.state) == 0:
            # Current player wins because the opponent removed the last digit(s)
            self.done = True
            reward = 1  # Current player wins
            return self._get_observation(), reward, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            reward = -10  # Penalty for invalid move
            return self._get_observation(), reward, True, False, {}

        # Perform the action
        if action <= 8:
            # Remove digits from the start
            num_digits = action + 1
            self.state = self.state[num_digits:]
        else:
            # Remove digits from the end
            num_digits = action - 9 + 1
            self.state = self.state[:-num_digits]

        # Check if N is empty after the move
        if len(self.state) == 0:
            # Current player loses because they removed the last digit(s)
            self.done = True
            reward = -1  # Current player loses
            return self._get_observation(), reward, True, False, {}

        # Switch to the next player
        self.current_player *= -1
        reward = 0  # No reward for a regular move
        return self._get_observation(), reward, False, False, {}

    def render(self):
        N_str = "".join(map(str, self.state))
        return f"Current number: {N_str}"

    def valid_moves(self):
        L = len(self.state)
        valid_actions = []
        # Actions to remove from the start
        for i in range(L):
            action = i  # Action indices from 0 to L-1
            valid_actions.append(action)
        # Actions to remove from the end
        for i in range(L):
            action = 9 + i  # Action indices from 9 to 9 + L - 1
            valid_actions.append(action)
        return valid_actions
