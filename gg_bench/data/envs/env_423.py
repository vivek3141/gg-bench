import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(
            9
        )  # Actions are numbers 1-9 (mapped from 0-8)
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, 1]
            ),  # Player 1 score, Player 2 score, current player (1 or 2)
            high=np.array([25, 25, 2]),
            dtype=np.int32,
        )

        # Game components
        self.prime_numbers = [2, 3, 5, 7]
        self.composite_numbers = [4, 6, 8, 9]
        self.unique_number = 1

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_scores = {1: 0, 2: 0}
        self.current_player = 1  # Player 1 starts the game
        self.done = False

        observation = np.array(
            [self.player_scores[1], self.player_scores[2], self.current_player],
            dtype=np.int32,
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                np.array(
                    [self.player_scores[1], self.player_scores[2], self.current_player],
                    dtype=np.int32,
                ),
                0,
                True,
                False,
                {},
            )

        number = action + 1  # Map action (0-8) to number (1-9)
        reward = 0

        # Determine target player and score adjustment based on the selected number
        if number in self.prime_numbers or number == self.unique_number:
            target_player = self.current_player
            added_score = number
        elif number in self.composite_numbers:
            target_player = 1 if self.current_player == 2 else 2  # Opponent
            added_score = number
        else:
            # Invalid number selected (should not occur)
            reward = -10
            terminated = False
            info = {}
            # Switch turn to the other player
            self.current_player = 1 if self.current_player == 2 else 2
            observation = np.array(
                [self.player_scores[1], self.player_scores[2], self.current_player],
                dtype=np.int32,
            )
            return observation, reward, terminated, False, info

        # Check if the move is valid (does not exceed 25 points)
        if self.player_scores[target_player] + added_score > 25:
            # Invalid move: Move is forfeited, switch turn
            reward = -10
            terminated = False
            info = {}
            # Switch turn to the other player
            self.current_player = 1 if self.current_player == 2 else 2
            observation = np.array(
                [self.player_scores[1], self.player_scores[2], self.current_player],
                dtype=np.int32,
            )
            return observation, reward, terminated, False, info
        else:
            # Valid move: Adjust scores accordingly
            self.player_scores[target_player] += added_score

            # Check for win condition
            if self.player_scores[self.current_player] == 25:
                # Current player wins
                reward = 1
                self.done = True
                terminated = True
                observation = np.array(
                    [self.player_scores[1], self.player_scores[2], self.current_player],
                    dtype=np.int32,
                )
                return observation, reward, terminated, False, {}
            elif self.player_scores[1 if self.current_player == 2 else 2] == 25:
                # Opponent wins (if current player added to opponent's score)
                reward = 0
                self.done = True
                terminated = True
                observation = np.array(
                    [self.player_scores[1], self.player_scores[2], self.current_player],
                    dtype=np.int32,
                )
                return observation, reward, terminated, False, {}
            else:
                # No win yet: Continue the game
                reward = 0
                terminated = False
                # Switch turn to the other player
                self.current_player = 1 if self.current_player == 2 else 2
                observation = np.array(
                    [self.player_scores[1], self.player_scores[2], self.current_player],
                    dtype=np.int32,
                )
                return observation, reward, terminated, False, {}

    def render(self):
        # Return a string representation of the current game state
        state_str = (
            f"Player 1 Score: {self.player_scores[1]}\n"
            f"Player 2 Score: {self.player_scores[2]}\n"
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        )
        return state_str

    def valid_moves(self):
        # Return a list of valid action indices based on the current game state
        valid_actions = []
        for action in range(9):
            number = action + 1
            if number in self.prime_numbers or number == self.unique_number:
                target_player = self.current_player
                added_score = number
            elif number in self.composite_numbers:
                target_player = 1 if self.current_player == 2 else 2  # Opponent
                added_score = number
            else:
                continue  # Invalid number, skip

            # Check if the move is valid
            if self.player_scores[target_player] + added_score <= 25:
                valid_actions.append(action)
        return valid_actions
