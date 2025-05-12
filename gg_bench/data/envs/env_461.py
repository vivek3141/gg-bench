import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions 0-8: Subtract Digit D (D = action + 1, since D ranges from 1 to 9)
        # Actions 9-18: Remove Digit D (D = action - 9, since D ranges from 0 to 9)
        self.action_space = spaces.Discrete(19)

        # Observation space: array of digits representing N, left-padded with zeros to a fixed size
        self.max_digits = 10  # Maximum number of digits to represent N
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(self.max_digits,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = 100  # Starting number N (>9)
        self.current_player = 1  # Current player (1 or -1)
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}  # Game already over

        reward = 0

        # Determine action type and corresponding digit D
        if 0 <= action <= 8:
            # Subtract Digit action
            D = action + 1  # Digits 1 to 9
            action_type = "subtract"
        elif 9 <= action <= 18:
            # Remove Digit action
            D = action - 9  # Digits 0 to 9
            action_type = "remove"
        else:
            # Invalid action index
            reward = -10
            self.done = True
            return self._get_obs(), reward, True, False, {}

        N_str = str(self.N)
        digits = [int(ch) for ch in N_str]

        # Perform the action if valid
        if action_type == "subtract":
            if D in digits:
                new_N = self.N - D
                if new_N > 0:
                    self.N = new_N
                else:
                    # Resulting N is not positive
                    reward = -10
                    self.done = True
                    return self._get_obs(), reward, True, False, {}
            else:
                # Digit D not in N
                reward = -10
                self.done = True
                return self._get_obs(), reward, True, False, {}
        elif action_type == "remove":
            if D in digits:
                index = digits.index(D)
                new_digits = digits[:index] + digits[index + 1 :]
                if len(new_digits) == 0:
                    # No digits left after removal
                    reward = -10
                    self.done = True
                    return self._get_obs(), reward, True, False, {}
                new_N_str = "".join(map(str, new_digits))
                if new_N_str[0] == "0":
                    # Leading zero after removal
                    reward = -10
                    self.done = True
                    return self._get_obs(), reward, True, False, {}
                self.N = int(new_N_str)
            else:
                # Digit D not in N
                reward = -10
                self.done = True
                return self._get_obs(), reward, True, False, {}

        # Check for win condition
        if 1 <= self.N <= 9:
            reward = 1
            self.done = True
            return self._get_obs(), reward, True, False, {}

        # Continue the game and switch players
        self.current_player *= -1
        return self._get_obs(), reward, False, False, {}

    def render(self):
        return f"Current N: {self.N}\n"

    def valid_moves(self):
        # Generate a list of valid action indices based on the current N
        valid_actions = []
        N_str = str(self.N)
        digits = [int(ch) for ch in N_str]

        # Check Subtract Digit actions
        for action in range(0, 9):
            D = action + 1
            if D in digits and (self.N - D) > 0:
                valid_actions.append(action)

        # Check Remove Digit actions
        for action in range(9, 19):
            D = action - 9
            if D in digits:
                index = digits.index(D)
                new_digits = digits[:index] + digits[index + 1 :]
                if len(new_digits) == 0:
                    continue  # Cannot have an empty N
                new_N_str = "".join(map(str, new_digits))
                if new_N_str[0] != "0":
                    valid_actions.append(action)

        return valid_actions

    def _get_obs(self):
        # Create an observation by padding the digits of N with zeros on the left
        N_str = str(self.N)
        digits = [int(ch) for ch in N_str]
        padded_digits = [0] * (self.max_digits - len(digits)) + digits
        return np.array(padded_digits, dtype=np.int32)
