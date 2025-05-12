import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 6 numbers (1-6) x 4 operations (+, -, *, /) = 24 possible actions
        self.action_space = spaces.Discrete(24)

        # Observation space: scores of both players [current player's score, opponent's score]
        # Each score ranges from 0 to 40, inclusive
        self.observation_space = spaces.Box(low=0, high=40, shape=(2,), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize scores for both players
        self.scores = [20, 20]  # Player 1 and Player 2 scores
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        return np.array(self.scores), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return np.array(self.scores), 0, True, {}, {}

        # Check if action is valid
        if action not in self.valid_moves():
            # Invalid move: Forfeit turn and assign penalty
            reward = -10
            self.current_player = 1 - self.current_player  # Switch player
            return np.array(self.scores), reward, self.done, {}

        # Map action to number selection `n` and operation
        n_index = action // 4
        n = n_index + 1  # `n` ranges from 1 to 6
        op_index = action % 4
        operations = ["+", "-", "*", "/"]
        operation = operations[op_index]

        # Get current and opponent player indices
        player = self.current_player
        opponent = 1 - self.current_player

        # Update self-score
        self.scores[player] -= n

        # Apply operation to opponent's score
        if operation == "+":
            self.scores[opponent] += n
        elif operation == "-":
            self.scores[opponent] -= n
        elif operation == "*":
            self.scores[opponent] *= n
        elif operation == "/":
            self.scores[opponent] //= n

        # Ensure opponent's score remains within valid range
        if self.scores[opponent] < 0 or self.scores[opponent] > 40:
            # Invalid move: Revert scores and forfeit turn
            self.scores[player] += n  # Revert self-score
            self.scores[opponent] = max(
                0, min(self.scores[opponent], 40)
            )  # Clamp score
            reward = -10
            self.current_player = opponent  # Switch player
            return np.array(self.scores), reward, self.done, {}

        # Check if current player has won
        if self.scores[player] == 0:
            self.done = True
            reward = 1  # Winning reward
            return np.array(self.scores), reward, self.done, {}

        # Switch to the next player
        self.current_player = opponent
        reward = 0
        return np.array(self.scores), reward, self.done, {}

    def render(self):
        board_str = f"Player {self.current_player + 1}'s turn\n"
        board_str += f"Player 1 Score: {self.scores[0]}\n"
        board_str += f"Player 2 Score: {self.scores[1]}"
        return board_str

    def valid_moves(self):
        valid_actions = []
        player_score = self.scores[self.current_player]
        opponent_score = self.scores[1 - self.current_player]
        for action in range(self.action_space.n):
            n_index = action // 4
            n = n_index + 1  # `n` ranges from 1 to 6
            op_index = action % 4
            operations = ["+", "-", "*", "/"]
            operation = operations[op_index]

            # Check self-score adjustment validity
            if player_score - n < 0:
                continue  # Invalid move: Negative self-score

            # Simulate opponent's score adjustment
            temp_opponent_score = opponent_score
            try:
                if operation == "+":
                    temp_opponent_score += n
                elif operation == "-":
                    temp_opponent_score -= n
                elif operation == "*":
                    temp_opponent_score *= n
                elif operation == "/":
                    if n == 0:
                        continue  # Invalid move: Division by zero
                    temp_opponent_score //= n
            except Exception:
                continue  # Invalid move due to exception

            # Check opponent's score validity
            if not isinstance(temp_opponent_score, int):
                continue  # Invalid move: Non-integer result
            if temp_opponent_score < 0 or temp_opponent_score > 40:
                continue  # Invalid move: Opponent's score out of bounds

            # Valid move
            valid_actions.append(action)
        return valid_actions
