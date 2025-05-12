import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers from 1 to 10 inclusive (actions 0-9 correspond to numbers 1-10)
        self.action_space = spaces.Discrete(10)

        # Define observation space:
        # Observation consists of [current player's score, opponent's score, opponent's last chosen number]
        # Scores range from 0 to 20, opponent's last number ranges from 0 (no move yet) to 10
        low = np.array([0, 0, 0], dtype=np.int32)
        high = np.array([20, 20, 10], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_scores = {1: 0, 2: 0}
        self.opponent_last_number = 0  # No last number at the beginning
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = self._get_observation()
        return observation, {}

    def _get_observation(self):
        # Returns observation: [current player's score, opponent's score, opponent's last chosen number]
        current_player_score = self.player_scores[self.current_player]
        opponent = 2 if self.current_player == 1 else 1
        opponent_score = self.player_scores[opponent]
        observation = np.array(
            [current_player_score, opponent_score, self.opponent_last_number],
            dtype=np.int32,
        )
        return observation

    def valid_moves(self):
        # Returns list of valid action indices for the current player
        opponent_last_number = self.opponent_last_number
        current_score = self.player_scores[self.current_player]
        valid_numbers = []
        for n in range(1, 11):
            if n != opponent_last_number and current_score + n <= 20:
                valid_numbers.append(n)
        valid_actions = [
            n - 1 for n in valid_numbers
        ]  # Convert numbers to action indices (0-9)
        return valid_actions

    def step(self, action):
        if self.done:
            # If the game is over, return current observation
            observation = self._get_observation()
            return observation, 0, True, False, {}

        # Check if current player has any valid moves
        valid_actions = self.valid_moves()
        if len(valid_actions) == 0:
            # Current player cannot make a valid move and loses the game
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Check if action is valid
        if action not in valid_actions:
            # Invalid move, player loses
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Process the action
        number_chosen = action + 1  # Convert action index to number (1-10)
        # Update current player's score
        self.player_scores[self.current_player] += number_chosen
        # Update opponent's last chosen number
        self.opponent_last_number = number_chosen

        # Check for winning condition
        if self.player_scores[self.current_player] == 20:
            reward = 1  # Current player wins
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Swap to the next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Prepare observation for the next player
        observation = self._get_observation()
        reward = 0
        return observation, reward, False, False, {}

    def render(self):
        # Return a string representation of the current game state
        lines = []
        lines.append(f"Player 1 Score: {self.player_scores[1]}")
        lines.append(f"Player 2 Score: {self.player_scores[2]}")
        lines.append(f"Opponent's last chosen number: {self.opponent_last_number}")
        lines.append(f"It's Player {self.current_player}'s turn.")
        return "\n".join(lines)
