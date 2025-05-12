import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to numbers 1 to 10 (action 0 corresponds to number 1)
        self.action_space = spaces.Discrete(10)

        # Observation space: [Player 1 Score, Player 2 Score, Current Player]
        # Current Player: 1 for Player 1's turn, -1 for Player 2's turn
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1], dtype=np.int32),
            high=np.array([50, 50, 1], dtype=np.int32),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1_score = 0
        self.player2_score = 0
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        observation = np.array(
            [self.player1_score, self.player2_score, self.current_player],
            dtype=np.int32,
        )
        return observation, {}

    def step(self, action):
        if self.done:
            return (
                np.array(
                    [self.player1_score, self.player2_score, self.current_player],
                    dtype=np.int32,
                ),
                0,
                True,
                False,
                {},
            )
        if action not in self.valid_moves():
            # Invalid move
            self.done = True
            reward = -10
            observation = np.array(
                [self.player1_score, self.player2_score, self.current_player],
                dtype=np.int32,
            )
            return observation, reward, True, False, {}
        number_chosen = action + 1  # Map action to number between 1 and 10
        if self.current_player == 1:
            self.player1_score += number_chosen
            # Prime Check and Doubling
            if self.is_prime(self.player1_score):
                self.player1_score *= 2
            # Win Condition Check
            if self.player1_score == 50:
                self.done = True
                reward = 1
                observation = np.array(
                    [self.player1_score, self.player2_score, self.current_player],
                    dtype=np.int32,
                )
                return observation, reward, True, False, {}
            # Overage Check
            if self.player1_score > 50:
                self.player1_score = 0
        elif self.current_player == -1:
            self.player2_score += number_chosen
            # Prime Check and Doubling
            if self.is_prime(self.player2_score):
                self.player2_score *= 2
            # Win Condition Check
            if self.player2_score == 50:
                self.done = True
                reward = 1
                observation = np.array(
                    [self.player1_score, self.player2_score, self.current_player],
                    dtype=np.int32,
                )
                return observation, reward, True, False, {}
            # Overage Check
            if self.player2_score > 50:
                self.player2_score = 0

        # Switch Current Player
        self.current_player *= -1

        # Prepare observation
        observation = np.array(
            [self.player1_score, self.player2_score, self.current_player],
            dtype=np.int32,
        )
        reward = 0
        terminated = False
        truncated = False
        return observation, reward, terminated, truncated, {}

    def render(self):
        state_str = f"Player 1 Score: {self.player1_score}\n"
        state_str += f"Player 2 Score: {self.player2_score}\n"
        if self.current_player == 1:
            state_str += "Player 1's turn.\n"
        else:
            state_str += "Player 2's turn.\n"
        return state_str

    def valid_moves(self):
        # Valid moves are numbers between 1 and 10 inclusive (mapped from actions 0-9)
        return list(range(10))

    def is_prime(self, n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        sqrt_n = int(np.sqrt(n)) + 1
        for i in range(3, sqrt_n, 2):
            if n % i == 0:
                return False
        return True
