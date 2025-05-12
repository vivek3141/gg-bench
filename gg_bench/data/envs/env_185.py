import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Numbers from 1 to 9
        self.numbers = np.arange(1, 10)

        # All possible ordered pairs where numbers are different
        self.pairs = [
            (i, j) for i in range(9) for j in range(9) if i != j
        ]  # Indices of numbers

        # Operations: 0 -> '+', 1 -> '-', 2 -> '*', 3 -> '/'
        self.operations = ["+", "-", "*", "/"]

        # Total actions: number of pairs * number of operations
        self.action_space = spaces.Discrete(len(self.pairs) * len(self.operations))

        # Observation space: 9 numbers availability (0 or 1), 2 scores (Player 1 and Player 2)
        # Numbers availability: 1 if available, 0 if used
        # Scores: can range from -100 to 100
        self.observation_space = spaces.Box(
            low=np.array([0] * 9 + [-100.0, -100.0]),
            high=np.array([1] * 9 + [100.0, 100.0]),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(
            9, dtype=np.float32
        )  # 1 if available, 0 if used
        self.player_scores = [0.0, 0.0]  # [Player 1 score, Player 2 score]
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        return self._get_obs(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        # Decode the action into number indices and operation
        total_operations = len(self.operations)
        pair_index = action // total_operations
        operation_index = action % total_operations

        if pair_index >= len(self.pairs):
            # Invalid action
            self.done = True
            return self._get_obs(), -10.0, True, False, {}

        number1_index, number2_index = self.pairs[pair_index]
        number1 = self.numbers[number1_index]
        number2 = self.numbers[number2_index]
        operation = self.operations[operation_index]

        # Check if numbers are available
        if (
            self.available_numbers[number1_index] == 0
            or self.available_numbers[number2_index] == 0
        ):
            # Invalid move: numbers not available
            self.done = True
            return self._get_obs(), -10.0, True, False, {}

        # Perform the operation
        if operation == "+":
            result = number1 + number2
        elif operation == "-":
            result = number1 - number2
        elif operation == "*":
            result = number1 * number2
        elif operation == "/":
            if number2 == 0:
                # Division by zero (should not happen with numbers 1-9)
                self.done = True
                return self._get_obs(), -10.0, True, False, {}
            result = number1 // number2  # Integer division
        else:
            # Invalid operation
            self.done = True
            return self._get_obs(), -10.0, True, False, {}

        # Update the player's score
        self.player_scores[self.current_player] += result

        # Remove used numbers from available numbers
        self.available_numbers[number1_index] = 0
        self.available_numbers[number2_index] = 0

        # Check for win/loss conditions
        if self.player_scores[self.current_player] == 50:
            # Player wins
            self.done = True
            return self._get_obs(), 1.0, True, False, {}
        elif self.player_scores[self.current_player] > 50:
            # Player loses
            self.done = True
            return self._get_obs(), 0.0, True, False, {}

        # Check if all numbers are used
        if np.sum(self.available_numbers) == 0:
            # Game ends
            self.done = True
            # Determine the winner (closest to 50 without exceeding)
            other_player = 1 - self.current_player
            self_score = self.player_scores[self.current_player]
            other_score = self.player_scores[other_player]
            self_diff = 50 - self_score if self_score <= 50 else float("inf")
            other_diff = 50 - other_score if other_score <= 50 else float("inf")
            if self_diff < other_diff:
                # Current player wins
                return self._get_obs(), 1.0, True, False, {}
            else:
                # Current player loses or draw
                return self._get_obs(), 0.0, True, False, {}

        # Switch to the next player
        self.current_player = 1 - self.current_player

        return self._get_obs(), 0.0, False, False, {}

    def _get_obs(self):
        # Observation consists of available numbers and player scores
        obs = np.concatenate(
            (self.available_numbers, np.array(self.player_scores, dtype=np.float32))
        )
        return obs

    def render(self):
        # Return a string representation of the game state
        available_numbers = [
            str(num)
            for idx, num in enumerate(self.numbers)
            if self.available_numbers[idx] == 1
        ]
        used_numbers = [
            str(num)
            for idx, num in enumerate(self.numbers)
            if self.available_numbers[idx] == 0
        ]
        render_str = f"Available Numbers: {' '.join(available_numbers)}\n"
        render_str += f"Used Numbers: {' '.join(used_numbers)}\n"
        render_str += f"Player 1 Score: {self.player_scores[0]}\n"
        render_str += f"Player 2 Score: {self.player_scores[1]}\n"
        render_str += f"Current Player: {'Player 1' if self.current_player == 0 else 'Player 2'}\n"
        return render_str

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        total_operations = len(self.operations)
        for pair_idx, (i, j) in enumerate(self.pairs):
            if self.available_numbers[i] == 1 and self.available_numbers[j] == 1:
                for op_idx in range(total_operations):
                    action = pair_idx * total_operations + op_idx
                    valid_actions.append(action)
        return valid_actions
