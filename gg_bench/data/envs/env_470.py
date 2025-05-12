import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: bids for both players
        # There are 25 possible combinations (5 bids for each player)
        self.action_space = spaces.Discrete(25)

        # Define observation space: scores for both players
        # Each player can have a score from 0 to 3
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 3]), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1_score = 0
        self.player2_score = 0
        self.done = False
        self.info = {}
        self.current_round = 1
        return self._get_obs(), self.info  # Return observation and info

    def step(self, action):
        if not self.action_space.contains(action) or self.done:
            return self._get_obs(), -10, True, False, self.info

        # Map action to bids for both players
        bid1 = (action // 5) + 1  # Player 1 bid
        bid2 = (action % 5) + 1  # Player 2 bid

        sum_bids = bid1 + bid2

        # Determine round winner
        if sum_bids <= 7:
            if bid1 > bid2:
                self.player1_score += 1
                round_winner = "Player 1"
            elif bid2 > bid1:
                self.player2_score += 1
                round_winner = "Player 2"
            else:
                round_winner = "Tie"
        else:  # sum_bids > 7
            if bid1 < bid2:
                self.player1_score += 1
                round_winner = "Player 1"
            elif bid2 < bid1:
                self.player2_score += 1
                round_winner = "Player 2"
            else:
                round_winner = "Tie"

        self.current_round += 1

        # Check for game over
        if self.player1_score >= 3:
            reward = 1
            self.done = True
        elif self.player2_score >= 3:
            reward = -1
            self.done = True
        else:
            reward = 0
            self.done = False

        # Update info for rendering
        self.info = {
            "bid1": bid1,
            "bid2": bid2,
            "sum_bids": sum_bids,
            "round_winner": round_winner,
        }

        return self._get_obs(), reward, self.done, False, self.info

    def render(self):
        output = f"Round {self.current_round - 1}\n"
        output += f"Player 1 Bid: {self.info.get('bid1', 'N/A')}\n"
        output += f"Player 2 Bid: {self.info.get('bid2', 'N/A')}\n"
        output += f"Sum of Bids: {self.info.get('sum_bids', 'N/A')}\n"
        output += f"Round Winner: {self.info.get('round_winner', 'N/A')}\n"
        output += f"Current Score - Player 1: {self.player1_score} | Player 2: {self.player2_score}\n"
        return output

    def valid_moves(self):
        return list(range(25))

    def _get_obs(self):
        return np.array([self.player1_score, self.player2_score], dtype=np.int32)
