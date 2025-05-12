import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomEnv(gym.Env):
    def __init__(self, N=50, N_max=100):
        super(CustomEnv, self).__init__()

        self.N = N  # Starting number
        self.N_max = N_max  # Maximum possible starting number

        # Define action space: numbers to subtract (0 to N_max)
        self.action_space = spaces.Discrete(self.N_max + 1)  # Actions: 0 to N_max

        # Define observation space: current N and previous number subtracted
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.N_max, self.N_max]),
            shape=(2,),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_N = self.N
        self.previous_subtracted = None
        self.done = False
        self.current_player = 1  # Not required for self-play but kept for clarity
        observation = np.array(
            [self.current_N, 0], dtype=np.int32
        )  # Use 0 if previous_subtracted is None
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            # Game has already ended
            observation = np.array(
                [self.current_N, self.previous_subtracted or 0], dtype=np.int32
            )
            reward = 0
            return observation, reward, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            observation = np.array(
                [self.current_N, self.previous_subtracted or 0], dtype=np.int32
            )
            return observation, reward, True, False, {}

        # Valid move
        subtract_number = (
            action  # Action corresponds directly to the number to subtract
        )
        self.current_N -= subtract_number
        self.previous_subtracted = subtract_number

        if self.current_N == 0:
            # Current player wins
            self.done = True
            reward = 1
            observation = np.array(
                [self.current_N, self.previous_subtracted], dtype=np.int32
            )
            return observation, reward, True, False, {}
        elif self.current_N < 0:
            # Should not occur if action is valid, but check for safety
            self.done = True
            reward = -10
            observation = np.array(
                [self.current_N, self.previous_subtracted], dtype=np.int32
            )
            return observation, reward, True, False, {}
        else:
            # Game continues
            reward = 0
            observation = np.array(
                [self.current_N, self.previous_subtracted], dtype=np.int32
            )
            return observation, reward, False, False, {}

    def valid_moves(self):
        if self.done:
            return []

        if self.previous_subtracted is None:
            # First move: any positive integer less than current N
            return [a for a in range(1, self.current_N)]
        else:
            # Subsequent moves: divisors or multiples of previous number subtracted
            previous = self.previous_subtracted
            valid_moves = []
            for a in range(1, self.current_N + 1):
                if previous % a == 0 or a % previous == 0:
                    valid_moves.append(a)
            return valid_moves

    def render(self):
        render_str = f"Current N: {self.current_N}\n"
        if self.previous_subtracted:
            render_str += f"Previous number subtracted: {self.previous_subtracted}\n"
        else:
            render_str += "First move.\n"
        return render_str
