import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the list of prime numbers under 30
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        # Action space: Selecting one of the 10 primes
        self.action_space = spaces.Discrete(len(self.primes))

        # Observation space: Player's score and opponent's score
        self.observation_space = spaces.Box(
            low=1, high=1000, shape=(2,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize scores and current player
        self.scores = {0: 1, 1: 1}  # Player 0 and Player 1
        self.current_player = 0  # Player 0 starts
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return self._get_obs(), 0, True, False, {}

        # Check if action is valid
        if action not in self.valid_moves():
            # Invalid move; player loses
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        # Perform the action
        prime = self.primes[action]
        new_score = self.scores[self.current_player] * prime
        self.scores[self.current_player] = new_score

        # Check for win/loss conditions
        if new_score == 1000:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_obs(), reward, True, False, {}
        elif new_score > 1000:
            # Current player loses
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}
        else:
            # Game continues; switch player
            reward = 0
            self.current_player = 1 - self.current_player
            return self._get_obs(), reward, False, False, {}

    def render(self):
        render_str = f"Current Player: Player {self.current_player + 1}\n"
        render_str += f"Player 1 Score: {self.scores[0]}\n"
        render_str += f"Player 2 Score: {self.scores[1]}\n"
        render_str += "Available Primes: " + ", ".join(map(str, self.primes)) + "\n"
        return render_str

    def valid_moves(self):
        # Return list of valid action indices (primes that won't cause the score to exceed 1000)
        valid_actions = []
        current_score = self.scores[self.current_player]
        for idx, prime in enumerate(self.primes):
            if current_score * prime <= 1000:
                valid_actions.append(idx)
        return valid_actions

    def _get_obs(self):
        # Observation: [current player's score, opponent's score]
        return np.array(
            [self.scores[self.current_player], self.scores[1 - self.current_player]],
            dtype=np.int32,
        )
