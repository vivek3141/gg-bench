import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Discrete actions from 0 to 19 corresponding to numbers 1 to 20
        self.action_space = spaces.Discrete(20)

        # Observation space: An array of 21 elements
        # Elements 0-19: values -1, 0, or 1 (opponent's selections, available numbers, current player's selections)
        # Element 20: last selected number scaled between -1 and 1
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(21,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the number state: 0 for available, 1 for current player's selections, -1 for opponent's selections
        self.number_state = np.zeros(20, dtype=np.int8)

        # Available numbers: numbers from 1 to 20
        self.available_numbers = list(range(1, 21))

        # Current player: 1 or -1
        self.current_player = 1

        # Last selected number: 0 at the start (no number selected yet)
        self.last_selected_number = 0

        # Game over flag
        self.done = False

        # Create initial observation
        observation = self._get_observation()

        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is over, no more actions can be taken
            return self._get_observation(), 0, True, False, {}

        selected_number = action + 1  # Map action to number
        if action < 0 or action >= 20 or self.number_state[action] != 0:
            # Invalid action: action out of bounds or number already selected
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Validate move according to game rules
        if self.last_selected_number == 0:
            # First move can be any number
            valid_move = True
        else:
            # Must select a number that is a divisor or multiple of the last selected number
            if (
                self.last_selected_number % selected_number == 0
                or selected_number % self.last_selected_number == 0
            ):
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            # Invalid move according to game rules
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move: update the game state
        self.number_state[action] = self.current_player  # Mark the number as selected
        self.available_numbers.remove(selected_number)
        self.last_selected_number = selected_number

        # Check for game over condition: does the next player have any valid moves?
        self.current_player *= -1  # Switch player
        valid_actions = self.valid_moves()

        if not valid_actions:
            # Next player has no valid moves: current player wins
            self.done = True
            reward = 1  # Current player wins
            # Switch back to the winning player for the observation
            self.current_player *= -1
            observation = self._get_observation()
            return observation, reward, True, False, {}
        else:
            # Game continues
            observation = self._get_observation()
            return observation, 0, False, False, {}

    def render(self):
        # Display the current state of the game
        output = "Current Player: {}\n".format(
            "Player 1" if self.current_player == 1 else "Player 2"
        )
        output += "Available Numbers: {}\n".format(
            [i + 1 for i in range(20) if self.number_state[i] == 0]
        )
        output += "Selected Numbers:\n"
        player1_numbers = [i + 1 for i in range(20) if self.number_state[i] == 1]
        player2_numbers = [i + 1 for i in range(20) if self.number_state[i] == -1]
        output += "  Player 1: {}\n".format(player1_numbers)
        output += "  Player 2: {}\n".format(player2_numbers)
        output += "Last Selected Number: {}\n".format(self.last_selected_number)
        return output

    def valid_moves(self):
        if self.last_selected_number == 0:
            # First move: any available number is valid
            return [i for i in range(20) if self.number_state[i] == 0]
        else:
            # Subsequent moves: numbers that are divisors or multiples of the last selected number
            valid_actions = []
            for action in range(20):
                if self.number_state[action] == 0:
                    number = action + 1
                    if (
                        number % self.last_selected_number == 0
                        or self.last_selected_number % number == 0
                    ):
                        valid_actions.append(action)
            return valid_actions

    def _get_observation(self):
        # Observation includes the number state and the last selected number scaled between -1 and 1
        observation = np.zeros(21, dtype=np.float32)
        # For number state, current player's selections are 1, opponent's are -1, available are 0
        observation[:20] = self.number_state * self.current_player
        # Scale last selected number between -1 and 1
        scaled_last_number = (self.last_selected_number / 20.0) * 2.0 - 1.0
        observation[20] = scaled_last_number
        return observation
