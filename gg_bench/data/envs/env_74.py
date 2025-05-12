import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 20 actions (numbers 1-10 with add or subtract)
        self.action_space = spaces.Discrete(20)

        # Define observation space:
        # [current_player_score, opponent_score, available_tokens(10 numbers), current_player_indicator]
        self.observation_space = spaces.Box(
            low=np.array([0, 0] + [0] * 10 + [0]),
            high=np.array([50, 50] + [2] * 10 + [1]),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_scores = [0, 0]  # Scores for player 0 and player 1
        self.available_tokens = np.array(
            [2] * 10, dtype=np.int32
        )  # Tokens for numbers 1-10
        self.current_player = 0  # Player 0 starts
        self.done = False
        observation = self.get_observation()
        return observation, {}

    def get_observation(self):
        # Observation: [current_player_score, opponent_score, available_tokens, current_player]
        opponent = 1 - self.current_player
        obs = np.array(
            [self.player_scores[self.current_player], self.player_scores[opponent]]
            + self.available_tokens.tolist()
            + [self.current_player],
            dtype=np.int32,
        )
        return obs

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        number_index = action // 2  # Index 0-9 for numbers 1-10
        operation = action % 2  # 0 for add, 1 for subtract
        number = number_index + 1  # Actual number (1-10)

        # Check if token is available
        if self.available_tokens[number_index] <= 0:
            self.done = True
            return self.get_observation(), -10, True, False, {}

        # Apply operation
        current_score = self.player_scores[self.current_player]
        if operation == 0:  # Add
            new_score = current_score + number
        else:  # Subtract
            new_score = current_score - number

        # Check if new score is valid
        if new_score < 0 or new_score > 50:
            self.done = True
            return self.get_observation(), -10, True, False, {}

        # Update game state
        self.player_scores[self.current_player] = new_score
        self.available_tokens[number_index] -= 1

        # Check for win condition
        if new_score == 50:
            self.done = True
            return self.get_observation(), 1, True, False, {}

        # Check if all tokens are used
        if np.sum(self.available_tokens) == 0:
            self.done = True
            opponent = 1 - self.current_player
            if self.player_scores[self.current_player] > self.player_scores[opponent]:
                return self.get_observation(), 1, True, False, {}
            elif (
                self.player_scores[self.current_player] == self.player_scores[opponent]
            ):
                return self.get_observation(), 0, True, False, {}
            else:
                return self.get_observation(), 0, True, False, {}

        # Switch player
        self.current_player = 1 - self.current_player
        observation = self.get_observation()
        return observation, 0, False, False, {}

    def render(self):
        output = f"--- Player {self.current_player + 1}'s Turn ---\n"
        output += "Available Numbers:\n"
        available_numbers = []
        for i in range(10):
            count = self.available_tokens[i]
            if count > 0:
                available_numbers.append(f"{i + 1}(x{count})")
        output += ", ".join(available_numbers) + "\n\n"
        output += f"Scores:\nPlayer 1: {self.player_scores[0]}\nPlayer 2: {self.player_scores[1]}\n"
        return output

    def valid_moves(self):
        valid_actions = []
        current_score = self.player_scores[self.current_player]
        for action in range(20):
            number_index = action // 2
            operation = action % 2
            number = number_index + 1

            if self.available_tokens[number_index] <= 0:
                continue  # Token not available

            if operation == 0:  # Add
                new_score = current_score + number
            else:  # Subtract
                new_score = current_score - number

            if 0 <= new_score <= 50:
                valid_actions.append(action)
        return valid_actions
