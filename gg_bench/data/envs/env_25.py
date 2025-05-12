import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_score=50):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 9 numbers * 2 operations = 18 discrete actions
        self.action_space = spaces.Discrete(18)

        # Observation space:
        # [current_player (1), player_scores (2), player_numbers (2 x 9)] = 21
        self.observation_space = spaces.Box(
            low=0,
            high=target_score,
            shape=(21,),
            dtype=np.int32,
        )

        self.target_score = target_score
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_scores = [0, 0]  # Scores for player 0 and player 1
        self.player_numbers = [
            np.ones(9, dtype=np.int32),
            np.ones(9, dtype=np.int32),
        ]  # Available numbers for each player
        self.current_player = 0  # Player 0 starts
        self.done = False
        return self._get_observation(), {}

    def _get_observation(self):
        # Observation format:
        # [current_player, player_scores[0], player_scores[1],
        #  player_numbers[0][0..8], player_numbers[1][0..8]]
        observation = np.concatenate(
            (
                np.array([self.current_player], dtype=np.int32),
                np.array(self.player_scores, dtype=np.int32),
                self.player_numbers[0],
                self.player_numbers[1],
            )
        )
        return observation

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}  # Game has ended

        # Map action to (number, operation)
        number, operation = self.action_index_to_number_operation(action)
        number_index = number - 1
        player_numbers = self.player_numbers[self.current_player]

        # Check if the chosen number is available
        if player_numbers[number_index] == 0:
            # Invalid move: number has already been used
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Apply the operation
        current_score = self.player_scores[self.current_player]
        if operation == "+":
            new_score = current_score + number
        elif operation == "*":
            if current_score == 0:
                new_score = 0  # 0 multiplied by any number is 0
            else:
                new_score = current_score * number
        else:
            raise ValueError("Invalid operation")

        # Update the player's numbers (mark the number as used)
        player_numbers[number_index] = 0

        # Check for overshooting the target
        if new_score > self.target_score:
            new_score = (
                self.target_score // 2
            )  # Reset to half the target number rounded down

        self.player_scores[self.current_player] = new_score

        # Check for win condition
        if new_score == self.target_score:
            reward = 1  # Current player wins
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Switch to the next player
        self.current_player = 1 - self.current_player

        # Check if the next player has valid moves
        if not self.valid_moves():
            # Skip turn if no valid moves
            self.current_player = 1 - self.current_player  # Switch back
            # Check if the current player has valid moves
            if not self.valid_moves():
                # No valid moves for both players; game ends in a draw
                self.done = True
                return self._get_observation(), 0, True, False, {}

        return self._get_observation(), 0, False, False, {}  # Continue the game

    def render(self):
        result = f"Current Player: {self.current_player}\n"
        result += f"Player 0 Score: {self.player_scores[0]}\n"
        result += f"Player 1 Score: {self.player_scores[1]}\n"
        result += "Player 0 Available Numbers: {}\n".format(
            [i + 1 for i, available in enumerate(self.player_numbers[0]) if available]
        )
        result += "Player 1 Available Numbers: {}\n".format(
            [i + 1 for i, available in enumerate(self.player_numbers[1]) if available]
        )
        return result

    def valid_moves(self):
        player_numbers = self.player_numbers[self.current_player]
        valid_actions = []
        for action in range(18):
            number, _ = self.action_index_to_number_operation(action)
            number_index = number - 1
            if player_numbers[number_index]:
                valid_actions.append(action)
        return valid_actions

    def action_index_to_number_operation(self, action):
        action_to_move = {
            0: (1, "+"),
            1: (1, "*"),
            2: (2, "+"),
            3: (2, "*"),
            4: (3, "+"),
            5: (3, "*"),
            6: (4, "+"),
            7: (4, "*"),
            8: (5, "+"),
            9: (5, "*"),
            10: (6, "+"),
            11: (6, "*"),
            12: (7, "+"),
            13: (7, "*"),
            14: (8, "+"),
            15: (8, "*"),
            16: (9, "+"),
            17: (9, "*"),
        }
        return action_to_move[action]
