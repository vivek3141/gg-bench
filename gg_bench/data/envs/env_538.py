import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Actions correspond to numbers 1-9
        self.observation_space = spaces.Box(low=0, high=9, shape=(9,), dtype=np.int32)

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.chain = np.zeros(9, dtype=np.int32)
        self.used_numbers = set()
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Choose starting number between 1 and 9
        if options and "starting_number" in options:
            starting_number = options["starting_number"]
            if starting_number < 1 or starting_number > 9:
                raise ValueError("Starting number must be between 1 and 9.")
        else:
            starting_number = self.np_random.integers(
                1, 10
            )  # Random integer between 1 and 9 inclusive

        self.chain[0] = starting_number
        self.chain_length = 1
        self.used_numbers.add(starting_number)

        return self.chain.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.chain.copy(), 0, True, False, {}

        move = action + 1  # Map action index to move number (1-9)

        last_number = self.chain[self.chain_length - 1]

        valid_moves = self.get_valid_moves(last_number, self.used_numbers)

        if action not in valid_moves:
            # Invalid move
            self.done = True
            return self.chain.copy(), -10, True, False, {}

        # Valid move, update chain
        self.chain[self.chain_length] = move
        self.chain_length += 1
        self.used_numbers.add(move)

        # Switch current player
        self.current_player *= -1

        # Check if the next player has valid moves
        next_valid_moves = self.get_valid_moves(move, self.used_numbers)
        if not next_valid_moves:
            # Current player wins
            self.done = True
            # Switch back to current player for consistency
            self.current_player *= -1
            return self.chain.copy(), 1, True, False, {}
        else:
            return self.chain.copy(), 0, False, False, {}

    def get_valid_moves(self, last_number, used_numbers):
        available_numbers = set(range(1, 10)) - used_numbers
        valid_moves = set()

        # One More: last_number + 1
        if last_number + 1 in available_numbers and 1 <= last_number + 1 <= 9:
            valid_moves.add(last_number + 1)

        # One Less: last_number - 1
        if last_number - 1 in available_numbers and 1 <= last_number - 1 <= 9:
            valid_moves.add(last_number - 1)

        # Double: last_number * 2
        if last_number * 2 in available_numbers and 1 <= last_number * 2 <= 9:
            valid_moves.add(last_number * 2)

        # Half: last_number / 2 (only if last_number is even)
        if last_number % 2 == 0:
            half = last_number // 2
            if half in available_numbers and 1 <= half <= 9:
                valid_moves.add(half)

        # Map numbers to action indices (numbers 1-9 correspond to actions 0-8)
        valid_actions = [num - 1 for num in valid_moves]
        return valid_actions

    def valid_moves(self):
        last_number = self.chain[self.chain_length - 1]
        return self.get_valid_moves(last_number, self.used_numbers)

    def render(self):
        chain_list = self.chain[: self.chain_length].tolist()
        available_numbers = sorted(set(range(1, 10)) - self.used_numbers)
        render_str = (
            f"Current Chain: {chain_list}\nAvailable Numbers: {available_numbers}\n"
        )
        return render_str
