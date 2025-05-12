import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Discrete(18) for numbers 1-9 with add or subtract operations
        self.action_space = spaces.Discrete(18)

        # Observation space: Availability of numbers (1-9) and scores of both players
        # First 9 elements: availability of numbers 1-9 (1.0 or 0.0)
        # Next 2 elements: current player's score and opponent's score (from -45 to +45)
        low_values = np.array([0] * 9 + [-45] * 2, dtype=np.float32)
        high_values = np.array([1] * 9 + [45] * 2, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_values, high=high_values, dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the shared sequence and player scores
        self.sequence_available = np.ones(
            9, dtype=np.float32
        )  # 1 if number is available, 0 if used
        self.player_scores = [0, 0]  # Player 1 and Player 2 scores
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False  # Game state indicator

        # Form the initial observation
        observation = np.concatenate(
            (
                self.sequence_available,
                [
                    self.player_scores[self.current_player],
                    self.player_scores[1 - self.current_player],
                ],
            )
        )
        info = {}
        return observation, info  # Return observation and info

    def valid_moves(self):
        # Returns a list of valid action indices based on available numbers
        valid_actions = []
        for n in range(9):  # Numbers 1 to 9
            if self.sequence_available[n] == 1:
                valid_actions.append(2 * n)  # Action to add number n+1
                valid_actions.append(2 * n + 1)  # Action to subtract number n+1
        return valid_actions

    def step(self, action):
        # Check for invalid action
        if action not in self.valid_moves():
            observation = np.concatenate(
                (
                    self.sequence_available,
                    [
                        self.player_scores[self.current_player],
                        self.player_scores[1 - self.current_player],
                    ],
                )
            )
            reward = -10
            self.done = True
            return observation, reward, self.done, False, {}

        # Decode the action into number and operation
        number_index = action // 2  # Index of the number (0 to 8)
        operation = action % 2  # 0 for add, 1 for subtract
        number = number_index + 1  # Actual number (1 to 9)

        # Update the current player's score
        if operation == 0:
            self.player_scores[self.current_player] += number
        else:
            self.player_scores[self.current_player] -= number

        # Mark the number as used
        self.sequence_available[number_index] = 0

        # Form the new observation
        observation = np.concatenate(
            (
                self.sequence_available,
                [
                    self.player_scores[self.current_player],
                    self.player_scores[1 - self.current_player],
                ],
            )
        )

        # Check for win condition
        if self.player_scores[self.current_player] == 0:
            reward = 1
            self.done = True
            return observation, reward, self.done, False, {}

        # Check if all numbers have been used
        if np.sum(self.sequence_available) == 0:
            self.done = True
            distance_current = abs(self.player_scores[self.current_player])
            distance_opponent = abs(self.player_scores[1 - self.current_player])

            # Determine winner based on closeness to zero or tiebreaker
            if distance_current < distance_opponent:
                reward = 1
            elif distance_current > distance_opponent:
                reward = -10
            else:
                reward = 1 if self.current_player == 1 else -10
            return observation, reward, self.done, False, {}

        # Continue the game and switch player
        reward = -10
        self.done = False
        self.current_player = 1 - self.current_player

        # Update the observation for the next player
        observation = np.concatenate(
            (
                self.sequence_available,
                [
                    self.player_scores[self.current_player],
                    self.player_scores[1 - self.current_player],
                ],
            )
        )
        return observation, reward, self.done, False, {}

    def render(self):
        # Generate a string representation of the game state
        sequence_numbers = [
            str(i + 1) for i in range(9) if self.sequence_available[i] == 1
        ]
        sequence_str = ", ".join(sequence_numbers)
        state_str = (
            f"Shared Sequence: {sequence_str}\n"
            f"Player 1 Score: {self.player_scores[0]}\n"
            f"Player 2 Score: {self.player_scores[1]}\n"
            f"Player {self.current_player + 1}'s Turn:"
        )
        return state_str
