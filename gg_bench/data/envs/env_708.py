import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: numbers from 1 to 10, represented as actions 0-9
        self.action_space = spaces.Discrete(10)

        # Observation space:
        # First 10 entries: number pool (1 if available, 0 if taken)
        # Entry 10: current number (0 if None, or 1-10)
        # Entry 11: current player's turn (-1 or 1)
        self.observation_space = spaces.Box(
            low=-1, high=10, shape=(12,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.number_pool = np.ones(10, dtype=np.int32)  # Numbers 1 to 10 are available
        self.current_number = 0  # None
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def _get_observation(self):
        # Construct the observation vector
        # First 10 entries: number pool status
        # Entry 10: current number
        # Entry 11: current player's turn
        observation = np.concatenate(
            (
                self.number_pool,
                np.array([self.current_number], dtype=np.int32),
                np.array([self.current_player], dtype=np.int32),
            )
        )
        return observation

    def step(self, action):
        if self.done:
            # Game is already over
            return self._get_observation(), -10, True, False, {}

        number = action + 1  # Map action 0-9 to number 1-10

        # Check if the selected number is available
        if self.number_pool[action] == 0:
            # Invalid move: number already taken
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check if the move is valid
        if self.current_number == 0:
            # First move: any number is valid
            valid_move = True
        else:
            # Subsequent moves: number must be a factor or multiple of the current number
            if (self.current_number % number == 0) or (
                number % self.current_number == 0
            ):
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            # Invalid move: not a factor or multiple
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move: update the game state
        self.number_pool[action] = 0  # Remove the number from the pool
        self.current_number = number  # Update the current number

        # Check if the next player has any valid moves
        next_player_can_move = False
        for i in range(10):
            if self.number_pool[i] == 1:
                next_number = i + 1
                if (self.current_number % next_number == 0) or (
                    next_number % self.current_number == 0
                ):
                    next_player_can_move = True
                    break

        if not next_player_can_move:
            # Next player cannot make a valid move: current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}
        else:
            # Switch to the next player
            self.current_player *= -1
            return self._get_observation(), 0, False, False, {}

    def render(self):
        # Generate a string representation of the current game state
        number_pool_list = [str(i + 1) for i in range(10) if self.number_pool[i] == 1]
        number_pool_str = ", ".join(number_pool_list)
        current_player_str = "Player 1" if self.current_player == 1 else "Player 2"
        current_number_str = self.current_number if self.current_number != 0 else "None"
        render_str = f"Current Player: {current_player_str}\n"
        render_str += f"Current Number: {current_number_str}\n"
        render_str += f"Available Numbers: {number_pool_str}\n"
        return render_str

    def valid_moves(self):
        # Return a list of valid action indices for the current player
        valid_moves = []
        for i in range(10):
            if self.number_pool[i] == 1:
                number = i + 1
                if self.current_number == 0:
                    # First move: all available numbers are valid
                    valid_moves.append(i)
                else:
                    # Subsequent moves: check if number is a factor or multiple
                    if (self.current_number % number == 0) or (
                        number % self.current_number == 0
                    ):
                        valid_moves.append(i)
        return valid_moves
