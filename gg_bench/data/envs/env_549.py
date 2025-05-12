import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 49 possible actions (numbers from 2 to 50 inclusive)
        self.action_space = spaces.Discrete(49)
        # Observation space consists of the number pool and the last opponent's number
        self.observation_space = spaces.Box(low=0, high=50, shape=(50,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All numbers from 2 to 50 are initially available (represented as 1)
        self.number_pool = np.ones(49, dtype=np.int8)
        self.last_opponent_number = 0  # No last opponent number at the start
        self.current_player = 1  # Player 1 starts
        self.done = False

        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is over
            return self._get_observation(), 0, True, False, {}

        selected_number = action + 2  # Map action index to number from 2 to 50

        if selected_number < 2 or selected_number > 50:
            # Action out of bounds
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check if the selected number is available
        if self.number_pool[action] == 0:
            # Number already taken
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check if the move is valid
        if self.last_opponent_number == 0:
            # First move, any number is valid
            valid_move = True
        else:
            # Move is valid if gcd > 1 with last opponent's number
            gcd = np.gcd(selected_number, self.last_opponent_number)
            if gcd > 1:
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move; update the game state
        self.number_pool[action] = 0  # Remove the number from the pool
        self.last_opponent_number = selected_number

        # Check if the opponent has any valid moves
        opponent_has_moves = False
        for i in range(49):
            if self.number_pool[i] == 1:
                number = i + 2
                gcd = np.gcd(number, self.last_opponent_number)
                if gcd > 1:
                    opponent_has_moves = True
                    break

        if not opponent_has_moves:
            # Opponent cannot make a valid move; current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        return self._get_observation(), 0, False, False, {}

    def render(self):
        # Return a visual representation of the game state as a string
        output = f"Current Player: Player {1 if self.current_player == 1 else 2}\n"
        output += "Numbers remaining in the pool:\n"
        numbers_in_pool = [
            str(i + 2) for i, val in enumerate(self.number_pool) if val == 1
        ]
        output += ", ".join(numbers_in_pool) + "\n"
        if self.last_opponent_number != 0:
            output += f"Last opponent's number: {self.last_opponent_number}\n"
        else:
            output += "No moves have been made yet.\n"
        return output

    def valid_moves(self):
        # Return a list of valid moves (indices in the action_space)
        valid_moves = []
        for i in range(49):
            if self.number_pool[i] == 1:
                number = i + 2
                if self.last_opponent_number == 0:
                    # Any number is valid on the first move
                    valid_moves.append(i)
                else:
                    gcd = np.gcd(number, self.last_opponent_number)
                    if gcd > 1:
                        valid_moves.append(i)
        return valid_moves

    def _get_observation(self):
        # Combine the number pool and last opponent's number into the observation
        observation = np.zeros(50, dtype=np.int8)
        observation[:49] = self.number_pool
        observation[49] = self.last_opponent_number
        return observation
