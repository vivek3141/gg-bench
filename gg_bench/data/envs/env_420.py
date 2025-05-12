import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.initial_shared_number = 100
        self.max_divisor = self.initial_shared_number
        self.max_actions = self.max_divisor - 1  # Divisors from 2 to 100

        # Define action space (action indices correspond to divisors from 2 to 100)
        self.action_space = spaces.Discrete(self.max_actions)

        # Observation space: [shared_number, used_divisors_flags]
        # shared_number ranges from 1 to initial_shared_number
        # used_divisors_flags is an array of binary flags for divisors from 2 to max_divisor (indices 0 to max_actions-1)
        self.observation_space = spaces.Box(
            low=np.array([1] + [0] * self.max_actions),
            high=np.array([self.initial_shared_number] + [1] * self.max_actions),
            dtype=np.int32,
        )

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.shared_number = self.initial_shared_number
        self.used_divisors = np.zeros(self.max_actions, dtype=np.int32)
        self.done = False
        self.current_player = 1  # Player 1 starts
        return self._get_obs(), {}  # Return observation and info

    def _get_obs(self):
        obs = np.concatenate(([self.shared_number], self.used_divisors))
        return obs

    def valid_moves(self):
        valid_moves = []
        for a in range(self.max_actions):
            divisor = a + 2  # Divisors are from 2 to max_divisor
            if self.used_divisors[a] == 0 and self.shared_number % divisor == 0:
                valid_moves.append(a)
        return valid_moves

    def step(self, action):
        if self.done:
            return self._get_obs(), -10, True, False, {}  # terminated is True

        divisor = action + 2  # Map action index to divisor

        # Check if action is valid
        if (
            action < 0
            or action >= self.max_actions
            or self.used_divisors[action] == 1
            or self.shared_number % divisor != 0
        ):
            # Invalid move
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}  # terminated is True

        # Valid move
        self.shared_number = self.shared_number // divisor
        self.used_divisors[action] = 1

        # Check if current player wins
        if self.shared_number == 1:
            self.done = True
            reward = +1
            return self._get_obs(), reward, True, False, {}  # terminated is True

        # Check if next player has valid moves
        valid_next_moves = self.valid_moves()
        if not valid_next_moves:
            # Next player cannot move, current player wins
            self.done = True
            reward = +1
            return self._get_obs(), reward, True, False, {}  # terminated is True

        # Switch player
        self.current_player = 3 - self.current_player  # Switch between player 1 and 2

        # Continue the game
        reward = 0
        return self._get_obs(), reward, False, False, {}  # terminated is False

    def render(self):
        used_divisors = np.where(self.used_divisors == 1)[0] + 2
        return (
            f"Shared Number: {self.shared_number}\nUsed divisors: {list(used_divisors)}"
        )

    def valid_moves(self):
        valid_moves = []
        for a in range(self.max_actions):
            divisor = a + 2  # Divisors are from 2 to max_divisor
            if self.used_divisors[a] == 0 and self.shared_number % divisor == 0:
                valid_moves.append(a)
        return valid_moves
