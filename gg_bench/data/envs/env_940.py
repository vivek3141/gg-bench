import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(20)
        # Observation space: 20 for numbers availability, 1 for required parity (-1.0, 0.0, 1.0)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(21,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All numbers from 1 to 20 are initially available
        self.available_numbers = np.ones(20, dtype=np.float32)
        self.current_player = 1  # Player 1 starts
        self.last_number_removed = None
        self.required_parity = 0.0  # No parity requirement on first move
        self.done = False
        self.truncated = False

        observation = self._get_observation()
        info = {}
        return observation, info  # Return observation and info

    def step(self, action):
        info = {}
        reward = 0

        if self.done:
            reward = 0
            return self._get_observation(), reward, self.done, self.truncated, info

        # Check if action is valid
        if not self._is_valid_action(action):
            # Invalid move
            reward = -10
            self.done = True
            return self._get_observation(), reward, self.done, self.truncated, info

        # Remove the selected number
        self.available_numbers[action] = 0
        number_removed = action + 1  # Numbers are from 1 to 20
        self.last_number_removed = number_removed

        # Update required parity for the next player
        if number_removed % 2 == 0:
            # Removed number is even; next player must pick an odd number
            self.required_parity = -1.0
        else:
            # Removed number is odd; next player must pick an even number
            self.required_parity = 1.0

        # Check if opponent has any valid moves
        if not self._has_valid_moves():
            # Current player wins
            reward = 1
            self.done = True
            return self._get_observation(), reward, self.done, self.truncated, info

        # Switch to the next player
        self.current_player *= -1
        reward = 0
        return self._get_observation(), reward, self.done, self.truncated, info

    def render(self):
        available_numbers = [
            str(i + 1) if self.available_numbers[i] == 1 else "X" for i in range(20)
        ]
        number_rows = [
            " ".join(available_numbers[i : i + 10]) for i in range(0, 20, 10)
        ]
        board_str = "\nAvailable Numbers:\n" + "\n".join(number_rows) + "\n"
        if self.last_number_removed is not None:
            parity_str = "even" if self.last_number_removed % 2 == 0 else "odd"
            last_move_str = (
                f"Last Number Removed: {self.last_number_removed} ({parity_str})"
            )
        else:
            last_move_str = "Last Number Removed: None"
        if self.required_parity == 0.0:
            parity_requirement = "You may select any number."
        elif self.required_parity == -1.0:
            parity_requirement = "You must select an odd number."
        else:
            parity_requirement = "You must select an even number."
        current_player_str = f"Player {1 if self.current_player == 1 else 2}'s Turn"
        render_str = (
            f"{board_str}\n{current_player_str}\n"
            f"----------------\n{last_move_str}\n{parity_requirement}\n"
        )
        return render_str

    def valid_moves(self):
        valid_moves = []
        for action in range(20):
            if self._is_valid_action(action):
                valid_moves.append(action)
        return valid_moves

    def _get_observation(self):
        observation = np.append(self.available_numbers, self.required_parity)
        return observation

    def _is_valid_action(self, action):
        if action < 0 or action >= 20:
            return False
        if self.available_numbers[action] == 0:
            return False
        number = action + 1  # Numbers are from 1 to 20
        # Check parity requirement
        if self.required_parity == 0.0:
            # No parity requirement on first move
            return True
        elif self.required_parity == -1.0:
            # Must pick an odd number
            return number % 2 != 0
        elif self.required_parity == 1.0:
            # Must pick an even number
            return number % 2 == 0
        else:
            # Should not reach here
            return False

    def _has_valid_moves(self):
        # Check if the opponent has any valid moves
        for action in range(20):
            if self.available_numbers[action] == 1:
                number = action + 1
                if self.required_parity == -1.0 and number % 2 != 0:
                    return True
                elif self.required_parity == 1.0 and number % 2 == 0:
                    return True
        return False
