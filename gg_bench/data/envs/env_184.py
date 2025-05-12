import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_score=50):
        super(CustomEnv, self).__init__()

        # Game parameters
        self.target_score = target_score

        # Define action and observation space
        # Actions are numbers from 1 to 9, mapped to indices 0 to 8
        self.action_space = spaces.Discrete(9)

        # Observation space:
        # [current_player_score, opponent_score, current_player_last_move, opponent_last_move]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.int32),
            high=np.array([self.target_score, self.target_score, 9, 9], dtype=np.int32),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.scores = np.array(
            [0, 0], dtype=np.int32
        )  # [Player 1 score, Player 2 score]
        self.last_moves = np.array(
            [0, 0], dtype=np.int32
        )  # [Player 1 last move, Player 2 last move]
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        self.winner = None
        observation = self._get_obs()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}  # Game is over

        if action not in range(9):
            # Invalid action
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {"reason": "Invalid action"}

        selected_number = action + 1  # Map action index to number 1-9
        opponent = 1 - self.current_player

        # Update current player's score
        self.scores[self.current_player] += selected_number

        # Apply multiples and factors rule
        # If not the first turn for each player
        if self.last_moves[opponent] != 0:
            last_opponent_number = self.last_moves[opponent]
            if selected_number % last_opponent_number == 0:
                # Selected number is a multiple of opponent's last number
                subtraction = last_opponent_number
                self.scores[opponent] = max(0, self.scores[opponent] - subtraction)
            elif last_opponent_number % selected_number == 0:
                # Selected number is a factor of opponent's last number
                subtraction = selected_number
                self.scores[opponent] = max(0, self.scores[opponent] - subtraction)

        # Update last move
        self.last_moves[self.current_player] = selected_number

        # Check for win condition
        reward = 0
        if self.scores[self.current_player] >= self.target_score:
            if self.scores[opponent] >= self.target_score:
                # Both players have reached or exceeded the target score
                if self.scores[self.current_player] > self.scores[opponent]:
                    # Current player wins
                    self.done = True
                    self.winner = self.current_player
                    reward = 1
                elif self.scores[self.current_player] == self.scores[opponent]:
                    # Tie, game continues until one player has higher score
                    pass  # Do nothing, game continues
                else:
                    # Opponent has higher score, opponent wins
                    self.done = True
                    self.winner = opponent
                    reward = 0  # Since current player lost
            else:
                # Current player wins
                self.done = True
                self.winner = self.current_player
                reward = 1
        elif self.scores[opponent] >= self.target_score:
            # Opponent has reached target score first
            self.done = True
            self.winner = opponent
            reward = 0  # Since current player lost

        # Switch players
        self.current_player = opponent

        return self._get_obs(), reward, self.done, False, {}

    def _get_obs(self):
        observation = np.array(
            [
                self.scores[self.current_player],
                self.scores[1 - self.current_player],
                self.last_moves[self.current_player],
                self.last_moves[1 - self.current_player],
            ],
            dtype=np.int32,
        )
        return observation

    def render(self):
        board = "--- Clash of Numbers ---\n"
        board += f"Target Score: {self.target_score}\n"
        board += f"Player 1 Score: {self.scores[0]}\n"
        board += f"Player 2 Score: {self.scores[1]}\n"
        board += f"Player {self.current_player +1}'s Turn\n"
        board += f"Last Moves - Player 1: {self.last_moves[0]}, Player 2: {self.last_moves[1]}\n"
        return board

    def valid_moves(self):
        return list(range(9))  # All actions from 0 to 8 are valid
