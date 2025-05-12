import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Discrete(6), actions 0-5 correspond to numbers 1-6
        self.action_space = spaces.Discrete(6)

        # Observation space:
        # [current_player_score, opponent_score, opponent_last_move]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), high=np.array([20, 20, 6]), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1  # Player 1 starts
        self.scores = [0, 0]  # Scores for Player 1 and Player 2
        self.last_moves = [0, 0]  # Last moves of Player 1 and Player 2
        self.done = False
        return self.get_observation(), {}  # Return observation and info

    def get_observation(self):
        player_idx = self.current_player - 1
        opponent_idx = 1 - player_idx
        current_player_score = self.scores[player_idx]
        opponent_score = self.scores[opponent_idx]
        opponent_last_move = self.last_moves[opponent_idx]
        observation = np.array(
            [current_player_score, opponent_score, opponent_last_move], dtype=np.int32
        )
        return observation

    def valid_moves(self):
        opponent_idx = 1 if self.current_player == 1 else 0
        opponent_last_move = self.last_moves[opponent_idx]
        possible_numbers = [1, 2, 3, 4, 5, 6]
        available_numbers = [
            num for num in possible_numbers if num != opponent_last_move
        ]
        player_idx = self.current_player - 1
        current_score = self.scores[player_idx]
        valid_numbers = [num for num in available_numbers if current_score + num <= 20]
        valid_actions = [num - 1 for num in valid_numbers]  # Convert to action indices
        return valid_actions

    def step(self, action):
        if self.done:
            # Game is over
            return self.get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()

        # Check if player has any valid moves
        if not valid_actions:
            # Current player cannot move; they lose
            self.done = True
            reward = -1  # Penalty for losing
            return self.get_observation(), reward, True, False, {}

        # Check if the action is valid
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            return self.get_observation(), reward, True, False, {}

        # Valid move
        number_chosen = action + 1  # Convert action index to number (1-6)
        player_idx = self.current_player - 1
        self.scores[player_idx] += number_chosen
        self.last_moves[player_idx] = number_chosen

        # Check for win condition
        if self.scores[player_idx] == 20:
            # Current player wins
            self.done = True
            reward = 1  # Reward for winning
            return self.get_observation(), reward, True, False, {}

        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1

        # Continue game
        return self.get_observation(), 0, False, False, {}

    def render(self):
        player_idx = self.current_player - 1
        opponent_idx = 1 - player_idx
        render_str = f"Current Player: {self.current_player}\n"
        render_str += f"Player {self.current_player} score: {self.scores[player_idx]}\n"
        render_str += f"Player {2 if self.current_player ==1 else 1} score: {self.scores[opponent_idx]}\n"
        render_str += f"Opponent's last move: {self.last_moves[opponent_idx]}\n"
        valid_numbers = [action + 1 for action in self.valid_moves()]
        render_str += f"Valid moves: {valid_numbers}\n"
        print(render_str)

    def valid_moves(self):
        opponent_idx = 1 if self.current_player == 1 else 0
        opponent_last_move = self.last_moves[opponent_idx]
        possible_numbers = [1, 2, 3, 4, 5, 6]
        available_numbers = [
            num for num in possible_numbers if num != opponent_last_move
        ]
        player_idx = self.current_player - 1
        current_score = self.scores[player_idx]
        valid_numbers = [num for num in available_numbers if current_score + num <= 20]
        valid_actions = [num - 1 for num in valid_numbers]  # Convert to action indices
        return valid_actions
