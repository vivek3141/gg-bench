import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(30)

        # Observation space:
        # Elements 0-29: Available numbers (1 if available, 0 if taken)
        # Element 30: Required parity for the next move (-1:any, 0:even, 1:odd)
        self.observation_space = spaces.Box(low=0, high=1, shape=(31,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the list of available numbers (1 to 30)
        self.available_numbers = np.ones(30, dtype=np.int8)
        # Initially, any number can be selected
        self.required_parity = -1  # -1 indicates no parity constraint
        self.done = False
        self.current_player = 1  # Player 1 starts
        # Build the initial observation
        observation = np.append(self.available_numbers, self.required_parity)
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        number = action + 1  # Convert action index to number (1-30)

        # Check if the action is valid
        if (
            action < 0
            or action >= 30
            or self.available_numbers[action] == 0
            or (self.required_parity != -1 and number % 2 != self.required_parity)
        ):
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move
        self.available_numbers[action] = 0  # Mark number as taken

        # Update required parity for the next move
        if number % 2 == 0:
            self.required_parity = 1  # Next number must be odd
        else:
            self.required_parity = 0  # Next number must be even

        # Check if the next player has any valid moves
        if not self._has_valid_moves():
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        return self._get_observation(), 0, False, False, {}

    def render(self):
        available_numbers = [
            str(i + 1) for i in range(30) if self.available_numbers[i] == 1
        ]
        taken_numbers = [
            str(i + 1) for i in range(30) if self.available_numbers[i] == 0
        ]
        parity_str = (
            "Any"
            if self.required_parity == -1
            else ("Even" if self.required_parity == 0 else "Odd")
        )
        render_str = f"Available Numbers: {', '.join(available_numbers)}\n"
        render_str += f"Taken Numbers: {', '.join(taken_numbers)}\n"
        render_str += f"Required Parity for Next Move: {parity_str}\n"
        render_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return render_str

    def valid_moves(self):
        valid_actions = []
        for action in range(30):
            number = action + 1
            if self.available_numbers[action] == 1:
                if self.required_parity == -1 or number % 2 == self.required_parity:
                    valid_actions.append(action)
        return valid_actions

    def _has_valid_moves(self):
        for action in range(30):
            number = action + 1
            if self.available_numbers[action] == 1 and (
                self.required_parity == -1 or number % 2 == self.required_parity
            ):
                return True
        return False

    def _get_observation(self):
        observation = np.append(self.available_numbers, self.required_parity)
        return observation
