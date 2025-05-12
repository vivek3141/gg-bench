import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: integers from 1 to 5 inclusive, represented as 0 to 4
        self.action_space = spaces.Discrete(
            5
        )  # actions 0 to 4 correspond to adding 1 to 5

        # Observation space: [total, player 0 score, player 1 score, current_player]
        # total from 0 to 30, scores from 0 to 50, current_player is 0 or 1
        low_obs = np.array([0, 0, 0, 0], dtype=np.int32)
        high_obs = np.array([30, 50, 50, 1], dtype=np.int32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total = 0
        self.scores = [0, 0]  # player 0 and player 1 scores
        self.current_player = 0  # 0 or 1
        self.done = False
        observation = np.array(
            [self.total, self.scores[0], self.scores[1], self.current_player],
            dtype=np.int32,
        )
        info = {}
        return observation, info

    def step(self, action):
        # Check if action is valid
        if action not in [0, 1, 2, 3, 4]:
            reward = -10
            self.done = True
            observation = np.array(
                [self.total, self.scores[0], self.scores[1], self.current_player],
                dtype=np.int32,
            )
            info = {}
            return observation, reward, self.done, False, info

        if self.done:
            reward = 0
            observation = np.array(
                [self.total, self.scores[0], self.scores[1], self.current_player],
                dtype=np.int32,
            )
            info = {}
            return observation, reward, self.done, False, info  # Game already over

        move = action + 1  # Convert action 0-4 to move 1-5

        # Update the total
        self.total += move

        # Scoring
        points = 0
        if self.total % 2 == 0:
            points += 1
        if self.total % 5 == 0:
            points += 2
        self.scores[self.current_player] += points

        # Check for game end condition
        if self.total >= 25:
            self.done = True

            # Determine winner
            if self.scores[0] > self.scores[1]:
                winner = 0
            elif self.scores[1] > self.scores[0]:
                winner = 1
            else:
                # If scores are tied, the last player to have taken a turn loses
                # So current player loses
                winner = 1 - self.current_player

            if winner == self.current_player:
                reward = 1  # Current player wins
            else:
                reward = 0  # Current player loses

            observation = np.array(
                [self.total, self.scores[0], self.scores[1], self.current_player],
                dtype=np.int32,
            )
            info = {}
            return observation, reward, self.done, False, info

        # Switch to next player
        self.current_player = 1 - self.current_player

        # Prepare observation
        observation = np.array(
            [self.total, self.scores[0], self.scores[1], self.current_player],
            dtype=np.int32,
        )
        reward = 0
        info = {}
        return observation, reward, self.done, False, info

    def render(self):
        output = f"Running Total: {self.total}\n"
        output += f"Player 0 Score: {self.scores[0]}\n"
        output += f"Player 1 Score: {self.scores[1]}\n"
        output += f"Current Player: {self.current_player}\n"
        return output

    def valid_moves(self):
        # Returns list of valid action indices (0 to 4)
        if self.done:
            return []
        else:
            return [0, 1, 2, 3, 4]  # Actions correspond to adding numbers 1 to 5
