import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: numbers from 1 to 15 (indices from 0 to 14)
        self.action_space = spaces.Discrete(15)

        # Observation space: 15 available numbers, 1 parity requirement
        # Available numbers: 1 if available, 0 if not
        # Parity requirement: 0 (none), 1 (must pick odd), 2 (must pick even)
        self.observation_space = spaces.Box(low=0, high=2, shape=(16,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize available numbers: numbers 1 to 15 (indices 0 to 14)
        self.numbers = np.ones(15, dtype=np.int8)
        # Initialize parity requirement: 0 (no requirement)
        self.parity_requirement = 0
        # Initialize done flag
        self.done = False
        # Observation
        observation = np.concatenate([self.numbers, [self.parity_requirement]])
        return observation, {}

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return (
                np.concatenate([self.numbers, [self.parity_requirement]]),
                0,
                True,
                {},
            )
        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            observation = np.concatenate([self.numbers, [self.parity_requirement]])
            return observation, -10, True, {}
        # Action is valid
        # Remove the number
        self.numbers[action] = 0
        # Update parity requirement for next player
        selected_number = action + 1  # Since action 0 corresponds to number 1
        if selected_number % 2 == 1:
            # Odd number selected
            self.parity_requirement = 1
        else:
            # Even number selected
            self.parity_requirement = 2
        # Check if game is over
        remaining_numbers = np.sum(self.numbers)
        if remaining_numbers == 1:
            # Game over, current player wins
            self.done = True
            observation = np.concatenate([self.numbers, [self.parity_requirement]])
            return observation, 1, True, {}
        # Continue the game
        observation = np.concatenate([self.numbers, [self.parity_requirement]])
        return observation, 0, False, {}

    def render(self):
        available_numbers = [str(i + 1) for i in range(15) if self.numbers[i] == 1]
        parity_str = {0: "None", 1: "Odd", 2: "Even"}
        state_str = f"Available numbers: {', '.join(available_numbers)}\n"
        state_str += f"Parity requirement: {parity_str[self.parity_requirement]}"
        return state_str

    def valid_moves(self):
        available_actions = [i for i in range(15) if self.numbers[i] == 1]
        if self.parity_requirement == 0:
            valid_actions = available_actions
        else:
            # Check if required parity numbers are available
            parity_numbers = [
                i
                for i in available_actions
                if (i + 1) % 2 == self.parity_requirement % 2
            ]
            if len(parity_numbers) == 0:
                # Parity requirement is lifted
                valid_actions = available_actions
            else:
                valid_actions = parity_numbers
        return valid_actions
