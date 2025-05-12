import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space corresponds to the numbers 1 to 10 (indexed from 0 to 9)
        self.action_space = spaces.Discrete(10)

        # Observation space:
        # Indexes 0-9: Player 1's available numbers (1 if available, 0 if used)
        # Indexes 10-19: Player 2's available numbers (1 if available, 0 if used)
        # Indexes 20-21: Players' scores (from 0 to 3)
        self.observation_space = spaces.Box(low=0, high=3, shape=(22,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Players' available numbers (sets of numbers from 1 to 10)
        self.available_numbers = [
            set(range(1, 11)),  # Player 1
            set(range(1, 11)),  # Player 2
        ]
        # Players' scores
        self.scores = [0, 0]  # [Player 1 score, Player 2 score]
        self.done = False
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            # If the game is over, no further actions are valid
            return self._get_observation(), 0, self.done, False, {}

        # Map action index to actual number (1-10)
        p1_num = action + 1

        if p1_num not in self.available_numbers[0]:
            # Invalid move by Player 1
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Remove chosen number from Player 1's available numbers
        self.available_numbers[0].remove(p1_num)

        # Simulate Player 2's action (randomly choose from available numbers)
        if len(self.available_numbers[1]) == 0:
            # Player 2 has no numbers left
            p2_num = None
        else:
            p2_num = np.random.choice(list(self.available_numbers[1]))
            self.available_numbers[1].remove(p2_num)

        # Resolve the round according to the game rules
        winner = None
        if p2_num is None:
            # Player 2 has no numbers left; Player 1 wins the round
            winner = 0
            result = "Player 2 has no numbers left. Player 1 wins the round!"
        else:
            # Apply game rules to determine the winner
            if p1_num == p2_num:
                result = "Tie"
                reward = 0
            elif p1_num == p2_num - 1:
                # Underdog Rule: Player 1 wins
                winner = 0
                result = "Underdog Rule Activated! Player 1 wins the round!"
            elif p2_num == p1_num - 1:
                # Underdog Rule: Player 2 wins
                winner = 1
                result = "Underdog Rule Activated! Player 2 wins the round!"
            else:
                if p1_num > p2_num:
                    winner = 0
                    result = "Player 1 wins the round!"
                else:
                    winner = 1
                    result = "Player 2 wins the round!"

        # Update scores and reward
        if winner is None:
            reward = 0  # Tie
        elif winner == 0:
            self.scores[0] += 1
            reward = 1  # Player 1 (agent) wins
        else:
            self.scores[1] += 1
            reward = 0  # Player 1 loses; no reward

        # Check for game termination condition
        if self.scores[0] >= 3 or self.scores[1] >= 3:
            self.done = True

        # Info dictionary can include details like the result and numbers chosen
        info = {"result": result, "player1_number": p1_num, "player2_number": p2_num}

        # Return the observation, reward, done flag, truncated flag, and info
        return self._get_observation(), reward, self.done, False, info

    def _get_observation(self):
        # Create an observation array containing players' available numbers and scores
        obs = np.zeros(22, dtype=np.int8)
        # Player 1's available numbers
        for i in range(10):
            obs[i] = 1 if (i + 1) in self.available_numbers[0] else 0
        # Player 2's available numbers
        for i in range(10):
            obs[i + 10] = 1 if (i + 1) in self.available_numbers[1] else 0
        # Players' scores
        obs[20] = self.scores[0]
        obs[21] = self.scores[1]
        return obs

    def render(self):
        # Return a string representation of the current game state
        s = "Current Game State:\n"
        s += f"Player 1's available numbers: {sorted(self.available_numbers[0])}\n"
        s += f"Player 2's available numbers: {sorted(self.available_numbers[1])}\n"
        s += f"Scores:\nPlayer 1: {self.scores[0]}\nPlayer 2: {self.scores[1]}\n"
        return s

    def valid_moves(self):
        # Return a list of valid action indices (numbers that Player 1 can choose)
        return [n - 1 for n in self.available_numbers[0]]
