import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 -> move 1 step, 1 -> move 2 steps, 2 -> move 3 steps
        self.action_space = spaces.Discrete(3)

        # Observation space:
        # [Number Path (9), Player 1 Position, Player 2 Position,
        #  Player 1 Score, Player 2 Score, Current Player Indicator]
        low = np.array([1] * 9 + [0, 0, 0, 0, 1], dtype=np.int32)
        high = np.array([9] * 9 + [9, 9, 100, 100, 2], dtype=np.int32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Generate a random permutation of numbers 1 to 9 for the number path
        self.number_path = np.random.permutation(np.arange(1, 10))
        # Positions start at 0 (before the first number)
        self.positions = {1: 0, 2: 0}
        # Scores start at 0
        self.scores = {1: 0, 2: 0}
        # Flags to indicate if players have finished
        self.finished = {1: False, 2: False}
        # Round number starts at 1
        self.round_number = 1
        # Current player (1 or 2)
        self.current_player = 1
        # Store the round when players finish
        self.finish_round = {1: None, 2: None}
        # Game over flag
        self.done = False
        return self._get_obs(), {}  # Observation and info

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Calculate steps (action 0 -> 1 step, action 1 -> 2 steps, etc.)
        steps = action + 1

        # Update the current player's position
        self.positions[self.current_player] += steps

        # Cap the position at the length of the number path
        if self.positions[self.current_player] > len(self.number_path):
            self.positions[self.current_player] = len(self.number_path)

        # Update the score if the player is still on the path
        # Positions are from 1 to 9 (indexing from 0)
        pos = self.positions[self.current_player]
        if pos <= len(self.number_path) and pos > 0:
            # Subtract 1 to get zero-based index
            self.scores[self.current_player] += self.number_path[pos - 1]

        # Check if the current player has reached or passed the end
        if self.positions[self.current_player] >= len(self.number_path):
            self.finished[self.current_player] = True
            # Record the round when the player finished
            self.finish_round[self.current_player] = self.round_number

        # Check if the game is over
        if self.finished[1] and self.finished[2]:
            # Both players have finished
            if self.finish_round[1] == self.finish_round[2]:
                # Both finished in the same round
                if self.scores[1] > self.scores[2]:
                    # Player 1 wins
                    winner = 1
                elif self.scores[1] < self.scores[2]:
                    # Player 2 wins
                    winner = 2
                else:
                    # No draws in Number Path; decide winner arbitrarily
                    winner = self.current_player
                self.done = True
                reward = 1 if self.current_player == winner else -1
            else:
                # Player who finished first wins
                if self.finish_round[1] < self.finish_round[2]:
                    winner = 1
                else:
                    winner = 2
                self.done = True
                reward = 1 if self.current_player == winner else -1
        elif self.finished[self.current_player]:
            # Current player finished first
            self.done = True
            reward = 1
        elif self.finished[1] or self.finished[2]:
            # Other player has finished earlier
            self.done = True
            reward = -1
        else:
            # Game continues
            reward = 0
            self.done = False

        # Switch current player
        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1
            # Increment round number after both players have moved
            self.round_number += 1

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        board_str = "Number Path:\n"
        path_str = ""
        for idx, num in enumerate(self.number_path):
            pos_str = f" {num} "
            for player_num, pos in self.positions.items():
                if pos - 1 == idx:
                    pos_str = f"P{player_num}({num})"
            path_str += pos_str + "-"

        path_str = path_str.rstrip("-")
        board_str += path_str + "\n"
        board_str += (
            f"Player 1 Position: {self.positions[1]}, Score: {self.scores[1]}\n"
        )
        board_str += (
            f"Player 2 Position: {self.positions[2]}, Score: {self.scores[2]}\n"
        )
        board_str += f"Current Player: Player {self.current_player}\n"
        return board_str

    def valid_moves(self):
        # From the current position, calculate valid actions
        c_pos = self.positions[self.current_player]
        max_steps = len(self.number_path) - c_pos
        valid_steps = [i for i in range(1, 4) if i <= max_steps and i > 0]
        # Convert steps to action indices
        valid_actions = [step - 1 for step in valid_steps]
        return valid_actions

    def _get_obs(self):
        # Create observation array
        obs = np.zeros(14, dtype=np.int32)
        # Number path
        obs[0:9] = self.number_path
        # Player positions
        obs[9] = self.positions[1]
        obs[10] = self.positions[2]
        # Player scores
        obs[11] = self.scores[1]
        obs[12] = self.scores[2]
        # Current player indicator
        obs[13] = self.current_player
        return obs
