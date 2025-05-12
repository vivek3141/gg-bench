import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(20)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(21,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.numbers_available = np.ones(20, dtype=np.int8)
        self.required_parity = -1  # -1 indicates no parity required (first turn)
        self.done = False
        observation = np.append(self.numbers_available, self.required_parity)
        return observation, {}  # Observation and info

    def step(self, action):
        if self.done:
            return (
                np.append(self.numbers_available, self.required_parity),
                0,
                True,
                False,
                {},
            )

        if action < 0 or action >= 20:
            return (
                np.append(self.numbers_available, self.required_parity),
                -10,
                True,
                False,
                {},
            )

        if self.numbers_available[action] == 0:
            return (
                np.append(self.numbers_available, self.required_parity),
                -10,
                True,
                False,
                {},
            )

        number_selected = action + 1
        number_parity = number_selected % 2  # 0 for even, 1 for odd

        # Check if the move matches the required parity
        if self.required_parity != -1 and number_parity != self.required_parity:
            return (
                np.append(self.numbers_available, self.required_parity),
                -10,
                True,
                False,
                {},
            )

        # Valid move, update the game state
        self.numbers_available[action] = 0

        # Update required parity for the next move
        if number_parity == 0:
            self.required_parity = 1  # Next required parity is odd
        else:
            self.required_parity = 0  # Next required parity is even

        # Check if the next player has any valid moves
        available_numbers = np.where(self.numbers_available == 1)[0] + 1
        valid_moves = [i for i in available_numbers if i % 2 == self.required_parity]

        if len(valid_moves) == 0:
            self.done = True
            return (
                np.append(self.numbers_available, self.required_parity),
                1,
                True,
                False,
                {},
            )

        return (
            np.append(self.numbers_available, self.required_parity),
            0,
            False,
            False,
            {},
        )

    def render(self):
        available_numbers = [
            str(i + 1) for i in range(20) if self.numbers_available[i] == 1
        ]
        state_str = "Available Numbers: " + ", ".join(available_numbers) + "\n"

        if self.required_parity == -1:
            parity_str = "No parity required (first turn)."
        elif self.required_parity == 0:
            parity_str = "Required parity: Even."
        else:
            parity_str = "Required parity: Odd."

        state_str += parity_str
        return state_str

    def valid_moves(self):
        if self.done:
            return []
        valid_moves = []
        for i in range(20):
            if self.numbers_available[i] == 1:
                number = i + 1
                number_parity = number % 2  # 0 for even, 1 for odd
                if self.required_parity == -1 or number_parity == self.required_parity:
                    valid_moves.append(i)
        return valid_moves
