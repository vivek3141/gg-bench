import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space:
        # 0-40: Declare prediction of final sum from -15 to 25 (prediction = action - 15)
        # 41-49: Select a number from available numbers (numbers 1 to 9)
        self.action_space = spaces.Discrete(50)

        # Observation space: a vector of 22 elements
        # Positions 0-8: Available numbers (1 if available, 0 if not)
        # Positions 9-17: Sequence of numbers selected so far (0 if not yet selected)
        # Positions 18-19: Predictions made by Player 1 and Player 2 (-99 if not yet made)
        # Position 20: Current player (1 for Player 1, -1 for Player 2)
        # Position 21: Total sum calculated as per game rules (alternating sum)
        self.observation_space = spaces.Box(
            low=-99, high=99, shape=(22,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.numbers_available = np.ones(9, dtype=np.int32)  # Numbers 1-9 available
        self.sequence = np.zeros(9, dtype=np.int32)  # Sequence of selected numbers
        self.position = 0  # Next position in the sequence
        self.predictions = {
            -1: -99,
            1: -99,
        }  # Predictions by both players (-99 if not made)
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.total_sum = 0  # Calculated as per game rules

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            reward = 0
            terminated = True
            truncated = False
            observation = self._get_observation()
            info = {}
            return observation, reward, terminated, truncated, info

        reward = -10  # Default reward for a valid move
        terminated = False
        truncated = False
        info = {}

        # Check if the action is a prediction
        if 0 <= action <= 40:
            prediction = action - 15  # Prediction range from -15 to 25
            if self.predictions[self.current_player] != -99:
                # Already made a prediction; invalid move
                reward = -100
                terminated = True
            else:
                # Record the prediction
                self.predictions[self.current_player] = prediction
        elif 41 <= action <= 49:
            number_index = action - 41  # Index in numbers 0-8 (numbers 1-9)
            if self.numbers_available[number_index] == 0:
                # Number already taken; invalid move
                reward = -100
                terminated = True
            else:
                # Valid number selection
                number = number_index + 1
                self.numbers_available[number_index] = 0
                self.sequence[self.position] = number
                self.position += 1
        else:
            # Invalid action
            reward = -100
            terminated = True

        # Check if game ends
        if self.position == 9 or np.sum(self.numbers_available) == 0:
            self.done = True
            # Calculate the final sum
            self.total_sum = self._calculate_total_sum()
            # Determine the winner
            winner = self._determine_winner()
            if winner == self.current_player:
                reward = 1
            else:
                reward = -1
            terminated = True

        # Switch player if the game is not over
        if not terminated:
            self.current_player *= -1

        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

    def render(self):
        board_str = "--- Number Quest Game State ---\n"
        board_str += "Available Numbers: "
        for i in range(9):
            if self.numbers_available[i] == 1:
                board_str += f"{i+1} "
        board_str += "\n"
        board_str += f"Shared Sequence: {self.sequence[:self.position]}\n"
        board_str += "Player Predictions:\n"
        for player in [1, -1]:
            pred = self.predictions[player]
            if pred != -99:
                board_str += f"  Player {1 if player == 1 else 2}: {pred}\n"
            else:
                board_str += f"  Player {1 if player == 1 else 2}: No prediction made\n"
        board_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        board_str += "-------------------------------\n"
        return board_str

    def valid_moves(self):
        valid_actions = []

        # If the player hasn't made a prediction yet, they can declare one
        if self.predictions[self.current_player] == -99:
            valid_actions.extend(range(0, 41))  # Predictions from -15 to 25

        # Add valid number selections
        for i in range(9):
            if self.numbers_available[i] == 1:
                valid_actions.append(i + 41)

        return valid_actions

    def _get_observation(self):
        obs = np.zeros(22, dtype=np.int32)
        obs[0:9] = self.numbers_available
        obs[9:18] = self.sequence
        obs[18] = self.predictions[1]
        obs[19] = self.predictions[-1]
        obs[20] = self.current_player
        obs[21] = self.total_sum
        return obs

    def _calculate_total_sum(self):
        total = 0
        sign = 1  # Start with positive sign
        for i in range(self.position):
            total += sign * self.sequence[i]
            sign *= -1  # Alternate the sign
        return total

    def _determine_winner(self):
        player1_pred = self.predictions[1]
        player2_pred = self.predictions[-1]
        actual_sum = self.total_sum

        player1_diff = abs(actual_sum - player1_pred) if player1_pred != -99 else None
        player2_diff = abs(actual_sum - player2_pred) if player2_pred != -99 else None

        if player1_pred == actual_sum and player2_pred != actual_sum:
            return 1  # Player 1 wins
        if player2_pred == actual_sum and player1_pred != actual_sum:
            return -1  # Player 2 wins
        if player1_pred == actual_sum and player2_pred == actual_sum:
            # Player who declared first wins
            if self._prediction_turn[1] < self._prediction_turn[-1]:
                return 1
            else:
                return -1
        if player1_diff is not None and player2_diff is not None:
            if player1_diff < player2_diff:
                return 1
            elif player2_diff < player1_diff:
                return -1
            else:
                # Tie in proximity; player who declared earlier loses
                if self._prediction_turn[1] < self._prediction_turn[-1]:
                    return -1
                else:
                    return 1
        elif player1_diff is not None:
            return 1
        elif player2_diff is not None:
            return -1
        else:
            # Neither player made a prediction; should not occur in a valid game
            return 0  # Draw (should not happen per game rules)

    def step(self, action):
        if self.done:
            reward = 0
            terminated = True
            truncated = False
            observation = self._get_observation()
            info = {}
            return observation, reward, terminated, truncated, info

        reward = -10  # Default reward for a valid move
        terminated = False
        truncated = False
        info = {}

        # Check if the action is a prediction
        if 0 <= action <= 40:
            prediction = action - 15  # Prediction range from -15 to 25
            if self.predictions[self.current_player] != -99:
                # Already made a prediction; invalid move
                reward = -100
                terminated = True
            else:
                # Record the prediction
                self.predictions[self.current_player] = prediction
                if not hasattr(self, "_prediction_turn"):
                    self._prediction_turn = {}
                self._prediction_turn[self.current_player] = self.position
        elif 41 <= action <= 49:
            number_index = action - 41  # Index in numbers 0-8 (numbers 1-9)
            if self.numbers_available[number_index] == 0:
                # Number already taken; invalid move
                reward = -100
                terminated = True
            else:
                # Valid number selection
                number = number_index + 1
                self.numbers_available[number_index] = 0
                self.sequence[self.position] = number
                self.position += 1
        else:
            # Invalid action
            reward = -100
            terminated = True

        # Check if game ends
        if self.position == 9 or np.sum(self.numbers_available) == 0:
            self.done = True
            # Calculate the final sum
            self.total_sum = self._calculate_total_sum()
            # Determine the winner
            winner = self._determine_winner()
            if winner == self.current_player:
                reward = 1
            else:
                reward = -1
            terminated = True

        # Switch player if the game is not over
        if not terminated:
            self.current_player *= -1

        observation = self._get_observation()
        return observation, reward, terminated, truncated, info
