import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to selecting numbers 1 to 20
        self.action_space = spaces.Discrete(20)

        # Observation space consists of:
        # - A binary vector indicating available numbers (1 for available, 0 for taken)
        # - The last selected number normalized between 0 and 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(21,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(20, dtype=np.float32)
        self.last_selected_number = 0.0  # No number selected yet
        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is over, no more moves can be made
            return self._get_observation(), 0.0, True, False, {}

        number = action + 1  # Convert action index to corresponding number (1-20)

        # Check if the selected number is available
        if self.available_numbers[action] != 1.0:
            # Invalid move: number already taken
            self.done = True
            reward = -10.0
            return self._get_observation(), reward, True, False, {}

        # Check if the move is valid according to game rules
        if self.last_selected_number == 0.0:
            # First move: any available number is valid
            valid_move = True
        else:
            last_number = int(self.last_selected_number)
            if (number % last_number == 0) or (last_number % number == 0):
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            # Invalid move according to game rules
            self.done = True
            reward = -10.0
            return self._get_observation(), reward, True, False, {}

        # Valid move: update the game state
        self.available_numbers[action] = 0.0  # Mark the number as taken
        self.last_selected_number = float(number)

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if the next player has any valid moves
        if len(self.valid_moves()) == 0:
            # The current player wins because the opponent cannot move
            self.done = True
            reward = 1.0
            winner = 2 if self.current_player == 1 else 1
            return self._get_observation(), reward, True, False, {"winner": winner}
        else:
            # The game continues
            reward = 0.0
            return self._get_observation(), reward, False, False, {}

    def render(self):
        # Generate a string representation of the game state
        available_numbers = [
            str(i + 1) for i in range(20) if self.available_numbers[i] == 1.0
        ]
        available_numbers_str = ", ".join(available_numbers)
        state_str = f"Available Numbers: {available_numbers_str}\n"

        if self.last_selected_number == 0.0:
            state_str += "No number has been selected yet.\n"
        else:
            state_str += f"Last selected number: {int(self.last_selected_number)}\n"

        state_str += f"Current player: Player {self.current_player}\n"
        return state_str

    def valid_moves(self):
        # Return a list of valid action indices based on the game rules
        valid_moves = []

        if self.last_selected_number == 0.0:
            # First move: all available numbers are valid
            valid_moves = [i for i in range(20) if self.available_numbers[i] == 1.0]
        else:
            last_number = int(self.last_selected_number)
            for i in range(20):
                if self.available_numbers[i] == 1.0:
                    number = i + 1
                    if (number % last_number == 0) or (last_number % number == 0):
                        valid_moves.append(i)

        return valid_moves

    def _get_observation(self):
        # Prepare the observation vector
        # Normalize the last selected number by dividing by 20
        observation = np.concatenate(
            (self.available_numbers, [self.last_selected_number / 20.0])
        )
        return observation
