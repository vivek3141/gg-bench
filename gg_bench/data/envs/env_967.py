import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Integers from 0 to 4 corresponding to adding numbers 1 to 5
        self.action_space = spaces.Discrete(5)

        # Observation space: Running total (integer between 0 and 35)
        self.observation_space = spaces.Box(low=0, high=35, shape=(1,), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Secret target number between 20 and 30 inclusive
        self.target_number = self.np_random.randint(20, 31)
        self.running_total = 0
        self.done = False
        return np.array([self.running_total], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # If the game is over, return the current state
            return np.array([self.running_total], dtype=np.int32), 0, True, False, {}

        # Map action index (0-4) to actual number (1-5)
        move = action + 1  # Actions are 0-4, moves are 1-5
        if move < 1 or move > 5:
            # Invalid move
            reward = -10
            self.done = True
            return (
                np.array([self.running_total], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Agent's move
        self.running_total += move

        # Check if agent loses
        if self.running_total > self.target_number:
            # Agent loses
            reward = -10
            self.done = True
            return (
                np.array([self.running_total], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        elif self.running_total == self.target_number:
            # Game continues
            pass  # Game continues
        else:
            # Game continues
            pass

        # Opponent's turn
        # Valid moves for opponent
        opponent_valid_moves = [
            i for i in range(1, 6) if self.running_total + i <= self.target_number + 5
        ]
        if len(opponent_valid_moves) == 0:
            # Opponent cannot make a move, agent wins
            reward = 1
            self.done = True
            return (
                np.array([self.running_total], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Opponent picks a random valid move
        opponent_move = self.np_random.choice(opponent_valid_moves)
        self.running_total += opponent_move

        # Check if opponent loses
        if self.running_total > self.target_number:
            # Opponent loses, agent wins
            reward = 1
            self.done = True
            return (
                np.array([self.running_total], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        elif self.running_total == self.target_number:
            # Game continues
            pass  # Game continues
        else:
            # Game continues
            pass

        # Game continues
        reward = 0
        return np.array([self.running_total], dtype=np.int32), reward, False, False, {}

    def render(self):
        return f"Running Total: {self.running_total}"

    def valid_moves(self):
        # Valid moves are integers between 1 and 5 inclusive that don't cause total to exceed maximum possible total
        valid_moves = []
        for i in range(5):  # Indices 0 to 4 correspond to moves 1 to 5
            move = i + 1
            if self.running_total + move <= 35:
                valid_moves.append(i)
        return valid_moves
