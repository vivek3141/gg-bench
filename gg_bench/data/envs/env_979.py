import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - Add 1, 1 - Subtract 1, 2 - Multiply by 2, 3 - Divide by 2
        self.action_space = spaces.Discrete(4)

        # Define observation space
        # 'current_number' is between 1 and 1000
        # 'numbers_used' is a binary vector indicating which numbers have been used (indices 0 to 1000)
        self.observation_space = spaces.Dict(
            {
                "current_number": spaces.Box(
                    low=1, high=1000, shape=(), dtype=np.int32
                ),
                "numbers_used": spaces.MultiBinary(1001),  # Indices 0 to 1000
            }
        )

        # Initialize variables
        self.current_number = None
        self.starting_number = None
        self.target_number = None
        self.numbers_used = None
        self.done = False

        # Initialize random number generator
        self.np_random = None
        self.seed()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Seeding
        if seed is not None:
            self.seed(seed)

        # Randomly select starting number S and target number T between 1 and 100, S != T
        self.starting_number = self.np_random.integers(1, 101)
        while True:
            self.target_number = self.np_random.integers(1, 101)
            if self.target_number != self.starting_number:
                break

        # Initialize current number and numbers used
        self.current_number = self.starting_number
        self.numbers_used = np.zeros(1001, dtype=np.int8)
        self.numbers_used[self.current_number] = 1  # Mark starting number as used
        self.done = False

        observation = {
            "current_number": np.array(self.current_number, dtype=np.int32),
            "numbers_used": self.numbers_used.copy(),
        }

        return observation, {}

    def step(self, action):
        if self.done:
            raise Exception("Game is over. Please reset the environment.")

        # Check if the action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            observation = {
                "current_number": np.array(self.current_number, dtype=np.int32),
                "numbers_used": self.numbers_used.copy(),
            }
            return observation, reward, True, False, {}

        # Apply the action to get the new current number
        if action == 0:  # Add 1
            new_current_number = self.current_number + 1
        elif action == 1:  # Subtract 1
            new_current_number = self.current_number - 1
        elif action == 2:  # Multiply by 2
            new_current_number = self.current_number * 2
        elif action == 3:  # Divide by 2
            new_current_number = self.current_number // 2  # Integer division

        # Check if new current number is valid and not repeated
        if (
            not (1 <= new_current_number <= 1000)
            or self.numbers_used[new_current_number] == 1
        ):
            # Invalid move (number out of bounds or repeated)
            self.done = True
            reward = -10
            observation = {
                "current_number": np.array(self.current_number, dtype=np.int32),
                "numbers_used": self.numbers_used.copy(),
            }
            return observation, reward, True, False, {}

        # Check for victory condition
        if new_current_number == self.target_number:
            # Player wins
            reward = 1
            self.done = True
        else:
            # Valid move, game continues
            reward = 0

        # Update the state
        self.current_number = new_current_number
        self.numbers_used[new_current_number] = 1  # Mark the number as used

        observation = {
            "current_number": np.array(self.current_number, dtype=np.int32),
            "numbers_used": self.numbers_used.copy(),
        }

        return observation, reward, self.done, False, {}

    def render(self):
        # Return a string representation of the current state
        used_numbers = np.where(self.numbers_used == 1)[0]
        state_str = (
            f"Current Number: {self.current_number}\n"
            f"Target Number: {self.target_number}\n"
            f"Numbers Used So Far: {used_numbers.tolist()}\n"
        )
        return state_str

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []

        # Action 0: Add 1
        new_number = self.current_number + 1
        if 1 <= new_number <= 1000 and self.numbers_used[new_number] == 0:
            valid_actions.append(0)

        # Action 1: Subtract 1
        new_number = self.current_number - 1
        if 1 <= new_number <= 1000 and self.numbers_used[new_number] == 0:
            valid_actions.append(1)

        # Action 2: Multiply by 2
        new_number = self.current_number * 2
        if (
            1 <= new_number <= 1000
            and new_number <= 1000
            and self.numbers_used[new_number] == 0
        ):
            valid_actions.append(2)

        # Action 3: Divide by 2 (only if current number is even)
        if self.current_number % 2 == 0:
            new_number = self.current_number // 2
            if 1 <= new_number <= 1000 and self.numbers_used[new_number] == 0:
                valid_actions.append(3)

        return valid_actions

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
