import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_divisor=7):
        super(CustomEnv, self).__init__()

        self.target_divisor = target_divisor
        self.action_space = spaces.Discrete(10)  # Digits from 0 to 9
        self.observation_space = spaces.Box(
            low=0, high=self.target_divisor - 1, shape=(1,), dtype=np.int32
        )

        self.current_modulo = None
        self.sequence = []
        self.done = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_modulo = 0
        self.sequence = []
        self.done = False
        return np.array([self.current_modulo], dtype=np.int32), {}  # observation, info

    def step(self, action):
        if self.done:
            raise RuntimeError("Game is over. Please reset the environment.")

        # Validate action
        if action < 0 or action > 9:
            raise ValueError("Invalid action. Action must be a digit between 0 and 9.")

        # Agent's move
        self.sequence.append(action)
        self.current_modulo = (self.current_modulo * 10 + action) % self.target_divisor

        if self.current_modulo == 0:
            # Agent loses
            self.done = True
            reward = 0  # Agent loses, receives no reward
            observation = np.array([self.current_modulo], dtype=np.int32)
            return observation, reward, self.done, False, {}

        # Opponent's move
        opponent_action = self.opponent_policy()
        self.sequence.append(opponent_action)
        self.current_modulo = (
            self.current_modulo * 10 + opponent_action
        ) % self.target_divisor

        if self.current_modulo == 0:
            # Opponent loses, agent wins
            self.done = True
            reward = 1  # Agent wins, receives positive reward
            observation = np.array([self.current_modulo], dtype=np.int32)
            return observation, reward, self.done, False, {}
        else:
            # Game continues
            reward = -10  # Penalty for valid move
            observation = np.array([self.current_modulo], dtype=np.int32)
            return observation, reward, self.done, False, {}

    def opponent_policy(self):
        # Opponent selects a random digit between 0 and 9
        return np.random.randint(0, 10)

    def render(self):
        seq_str = "".join(map(str, self.sequence))
        return f"Current Sequence: {seq_str} | Current Modulo: {self.current_modulo}"

    def valid_moves(self):
        # All digits from 0 to 9 are valid moves
        return list(range(10))
