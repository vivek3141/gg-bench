import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=10):
        super(CustomEnv, self).__init__()

        self.N = N
        self.numbers = np.arange(2, N + 1)  # [2, 3, ..., N]
        self.num_actions = N - 1  # Since numbers are from 2 to N inclusive
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.num_actions,), dtype=np.int32
        )  # 0: Available, 1: Locked, 2: Removed
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(
            self.num_actions, dtype=np.int32
        )  # All numbers start as available
        self.current_player = 1  # Player 1 starts
        return self.state.copy(), {}  # Return observation and info

    def valid_moves(self):
        return [i for i in range(self.num_actions) if self.state[i] == 0]

    def step(self, action):
        if action not in self.valid_moves():
            # Invalid move
            return (
                self.state.copy(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Remove the selected number
        self.state[action] = 2  # Mark as removed
        number_removed = self.numbers[action]

        # Lock multiples and factors of the removed number (excluding the removed number itself)
        for i in range(self.num_actions):
            if self.state[i] == 0:
                num = self.numbers[i]
                if num != number_removed and (
                    number_removed % num == 0 or num % number_removed == 0
                ):
                    self.state[i] = 1  # Mark as locked

        # Check if any available numbers remain
        if not any(self.state == 0):
            # Current player wins
            return self.state.copy(), 1, True, False, {}
        else:
            # Switch to the next player
            self.current_player *= -1
            return self.state.copy(), 0, False, False, {}

    def render(self):
        render_str = f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        render_str += "Numbers State:\n"
        for i in range(self.num_actions):
            num = self.numbers[i]
            state = self.state[i]
            if state == 0:
                state_str = "Available"
            elif state == 1:
                state_str = "Locked"
            else:
                state_str = "Removed"
            render_str += f"{num}: {state_str}\n"
        return render_str

    def valid_moves(self):
        return [i for i in range(self.num_actions) if self.state[i] == 0]
