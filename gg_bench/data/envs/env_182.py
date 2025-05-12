import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=20, target_score=30):
        super(CustomEnv, self).__init__()

        self.N = N  # Maximum number in the sequence
        self.target_score = target_score  # Target score to win the game

        # Define action and observation space
        # The agent selects an action from 0 to N-1, corresponding to numbers 1 to N
        self.action_space = spaces.Discrete(self.N)
        # Observation space includes:
        # - N elements indicating the availability of numbers 1 to N (1 if available, 0 if taken)
        # - Last opponent's number selected (normalized between 0 and 1)
        # - Current player's score (normalized between 0 and 1)
        # - Opponent's score (normalized between 0 and 1)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.N + 3,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the sequence of available numbers (1 if available, 0 if taken)
        self.available_numbers = np.ones(self.N, dtype=np.float32)
        # Initialize player scores
        self.player_scores = {1: 0, 2: 0}
        # Set the current player (Player 1 starts)
        self.current_player = 1
        # Initialize the last number selected by the opponent (0 indicates any number is valid)
        self.last_opponent_number = 0  # No last opponent's number at the start
        self.done = False

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            # If the game is already over, return the current state with no changes
            observation = self._get_observation()
            reward = 0
            info = {}
            return observation, reward, True, False, info

        valid_moves = self.valid_moves()
        # Check if there are valid moves available
        if not valid_moves:
            # No valid moves: skip the turn
            self._skip_turn()
            observation = self._get_observation()
            reward = 0
            info = {}
            return observation, reward, False, False, info

        number_selected = action + 1  # Map action to number (1 to N)

        # Check if the selected number is available
        if self.available_numbers[action] != 1.0:
            # Invalid move: number is already taken
            observation = self._get_observation()
            reward = -10
            self.done = True
            info = {"error": "Selected number already taken"}
            return observation, reward, True, False, info

        # Check if the selected number is valid according to the game rules
        if not self._is_valid_move(number_selected):
            # Invalid move: does not satisfy the multiple/factor rule
            observation = self._get_observation()
            reward = -10
            self.done = True
            info = {"error": "Invalid move according to game rules"}
            return observation, reward, True, False, info

        # Valid move: update the game state
        # Update the player's score
        self.player_scores[self.current_player] += number_selected
        # Remove the selected number from the available numbers
        self.available_numbers[action] = 0.0
        # Update the last opponent's number
        self.last_opponent_number = number_selected

        # Check for victory condition
        if self.player_scores[self.current_player] >= self.target_score:
            observation = self._get_observation()
            reward = 1
            self.done = True
            info = {"winner": self.current_player}
            return observation, reward, True, False, info

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Prepare the observation for the next player
        observation = self._get_observation()
        reward = 0
        info = {}
        return observation, reward, False, False, info

    def render(self):
        # Generate a string representation of the game state
        sequence_str = "Available Numbers: " + str(
            [i + 1 for i in range(self.N) if self.available_numbers[i] == 1.0]
        )
        score_str = f"Scores -> Player 1: {self.player_scores[1]} | Player 2: {self.player_scores[2]}"
        turn_str = f"Player {self.current_player}'s turn."
        last_num_str = f"Last number selected by opponent: {self.last_opponent_number}"
        return f"{sequence_str}\n{score_str}\n{turn_str}\n{last_num_str}"

    def valid_moves(self):
        # Return a list of valid actions (indices) for the current player
        valid_actions = []
        if self.last_opponent_number == 0:
            # First move: all available numbers are valid
            valid_actions = [
                i for i in range(self.N) if self.available_numbers[i] == 1.0
            ]
        else:
            for i in range(self.N):
                if self.available_numbers[i] == 1.0:
                    number = i + 1
                    if self._is_multiple_or_factor(number, self.last_opponent_number):
                        valid_actions.append(i)
        return valid_actions

    def _is_valid_move(self, number_selected):
        if self.last_opponent_number == 0:
            # First move: any number is valid
            return True
        else:
            return self._is_multiple_or_factor(
                number_selected, self.last_opponent_number
            )

    @staticmethod
    def _is_multiple_or_factor(a, b):
        # Check if a is a multiple or factor of b
        return a % b == 0 or b % a == 0

    def _skip_turn(self):
        # Skip the current player's turn due to no valid moves
        # Update the last opponent's number to remain the same
        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1

    def _get_observation(self):
        # Prepare the observation array
        observation = np.zeros(self.N + 3, dtype=np.float32)
        # Indicate availability of numbers (1 if available, 0 if taken)
        observation[: self.N] = self.available_numbers
        # Normalize the last opponent's number between 0 and 1
        observation[self.N] = self.last_opponent_number / self.N
        # Normalize the current player's score
        observation[self.N + 1] = (
            self.player_scores[self.current_player] / self.target_score
        )
        # Normalize the opponent's score
        opponent = 2 if self.current_player == 1 else 1
        observation[self.N + 2] = self.player_scores[opponent] / self.target_score
        return observation
