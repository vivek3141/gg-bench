import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sympy import isprime


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the starting number
        self.starting_number = 60

        # Compute all proper divisors of the starting number (excluding 1 and itself)
        self.all_divisors = self.get_proper_divisors(self.starting_number)

        # Map action indices to divisors and vice versa
        self.action_to_divisor = {
            index: divisor for index, divisor in enumerate(self.all_divisors)
        }
        self.divisor_to_action = {
            divisor: index for index, divisor in enumerate(self.all_divisors)
        }

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.all_divisors))
        # Observation: [shared_number] + [used_divisors]
        self.observation_space = spaces.Box(
            low=np.array([2] + [0] * len(self.all_divisors)),
            high=np.array([self.starting_number] + [1] * len(self.all_divisors)),
            dtype=np.int32,
        )

        # Initialize the environment
        self.reset()

    def get_proper_divisors(self, n):
        divisors = []
        for i in range(2, n):
            if n % i == 0:
                divisors.append(i)
        return divisors

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = self.starting_number
        self.used_divisors = np.zeros(len(self.all_divisors), dtype=np.int32)
        self.current_player = 1
        self.done = False
        observation = self.get_observation()
        return observation, {}

    def get_observation(self):
        return np.array(
            [self.shared_number] + self.used_divisors.tolist(), dtype=np.int32
        )

    def valid_moves(self):
        valid_actions = []
        for index, divisor in self.action_to_divisor.items():
            if (
                self.used_divisors[index] == 0
                and self.shared_number % divisor == 0
                and divisor != self.shared_number
            ):
                valid_actions.append(index)
        return valid_actions

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}
        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return (
                self.get_observation(),
                -10,
                True,
                False,
                {},
            )  # Reward, terminated, truncated, info

        # Perform the action
        divisor = self.action_to_divisor[action]
        self.shared_number = int(self.shared_number / divisor)
        self.used_divisors[action] = 1

        # Check for win condition
        if isprime(self.shared_number):
            self.done = True
            return (
                self.get_observation(),
                1,
                True,
                False,
                {},
            )  # Reward, terminated, truncated, info

        # Check if any valid moves remain
        if len(self.valid_moves()) == 0:
            # No valid moves remain, but the game continues until a player cannot move
            pass

        # Switch current player
        self.current_player *= -1

        # Return observation, reward, done, info
        return self.get_observation(), 0, False, False, {}

    def render(self):
        state_str = f"Shared Number: {self.shared_number}\n"
        used_divisors = [
            divisor
            for index, divisor in self.action_to_divisor.items()
            if self.used_divisors[index] == 1
        ]
        available_divisors = [
            divisor
            for index, divisor in self.action_to_divisor.items()
            if self.used_divisors[index] == 0 and self.shared_number % divisor == 0
        ]
        state_str += f"Available Divisors: {', '.join(map(str, available_divisors))}\n"
        state_str += f"Divisors Used: {', '.join(map(str, used_divisors))}\n"
        state_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return state_str
