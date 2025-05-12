import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: integers from 0 to 4 (maps to subtraction numbers 1 to 5)
        self.action_space = spaces.Discrete(5)
        # Observation space: [my_score, opponent_score, last_opponent_action]
        # my_score and opponent_score in [0, 20], last_opponent_action in [-1, 5]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1]), high=np.array([20, 20, 5]), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_scores = {1: 20, 2: 20}
        self.last_actions = {1: -1, 2: -1}
        self.current_player = 1
        self.done = False

        # Return the initial observation
        observation = np.array(
            [
                self.player_scores[self.current_player],
                self.player_scores[self._opponent()],
                self.last_actions[self._opponent()],
            ],
            dtype=np.int32,
        )

        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map action index to subtraction number (1 to 5)
        subtraction_number = action + 1

        # Validate the action
        valid = True
        info = {}

        # Check if subtraction_number is the same as opponent's last action
        if subtraction_number == self.last_actions[self._opponent()]:
            valid = False
            info["reason"] = (
                "Cannot subtract the same number the opponent just subtracted."
            )

        # Check if subtracting the number would reduce score below zero
        if self.player_scores[self.current_player] - subtraction_number < 0:
            valid = False
            info["reason"] = "Subtraction would reduce score below zero."

        # If action is invalid
        if not valid:
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, info

        # Apply the subtraction
        self.player_scores[self.current_player] -= subtraction_number
        self.last_actions[self.current_player] = subtraction_number

        # Check for win condition
        if self.player_scores[self.current_player] == 0:
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Check if the next player has any valid moves
        self.current_player = self._opponent()
        if len(self.valid_moves()) == 0:
            # Next player has no valid moves, current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, {}

        # No reward for a valid move that doesn't end the game
        reward = 0

        return self._get_observation(), reward, False, False, {}

    def render(self):
        state_str = f"Player {self.current_player}'s turn.\n"
        state_str += f"Player 1 Score: {self.player_scores[1]}\n"
        state_str += f"Player 2 Score: {self.player_scores[2]}\n"
        state_str += f"Last action of opponent (Player {self._opponent()}): {self.last_actions[self._opponent()]}\n"
        return state_str

    def valid_moves(self):
        # Return list of valid moves (action indices: 0 to 4) for the current player
        valid_moves = []
        for i in range(5):
            subtraction_number = i + 1
            if subtraction_number == self.last_actions[self._opponent()]:
                continue
            if self.player_scores[self.current_player] - subtraction_number < 0:
                continue
            valid_moves.append(i)
        return valid_moves

    def _get_observation(self):
        observation = np.array(
            [
                self.player_scores[self.current_player],
                self.player_scores[self._opponent()],
                self.last_actions[self._opponent()],
            ],
            dtype=np.int32,
        )
        return observation

    def _opponent(self):
        return 2 if self.current_player == 1 else 1
