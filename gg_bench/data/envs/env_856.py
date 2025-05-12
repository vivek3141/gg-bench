import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 14 discrete actions (7 indices * 2 operations)
        self.action_space = spaces.Discrete(14)
        # Observation space: An array of 7 integers between 1 and 9 inclusive
        self.observation_space = spaces.Box(low=1, high=9, shape=(7,), dtype=np.int32)

        # Assigned parity: Player 1 is even (0), Player 2 is odd (1)
        self.assigned_parity = {1: 0, 2: 1}

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the array with random integers between 1 and 9 inclusive
        self.array = np.random.randint(1, 10, size=7, dtype=np.int32)
        # Set the current player (1 or 2)
        self.current_player = 1
        self.done = False
        return self.array.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.array.copy(), 0, True, False, {}

        # Map the action to index and operation
        index = action // 2
        op = "+" if action % 2 == 0 else "-"

        # Check if the action is valid
        if index < 0 or index >= 7:
            return self.array.copy(), -10, True, False, {}

        current_value = self.array[index]

        if op == "+":
            if current_value >= 9:
                return self.array.copy(), -10, True, False, {}
            else:
                self.array[index] += 1
        elif op == "-":
            if current_value <= 1:
                return self.array.copy(), -10, True, False, {}
            else:
                self.array[index] -= 1
        else:
            return self.array.copy(), -10, True, False, {}

        # Check for win condition
        parity = self.assigned_parity[self.current_player]
        if np.all(self.array % 2 == parity):
            self.done = True
            return self.array.copy(), 1, True, False, {}

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1
        return self.array.copy(), 0, False, False, {}

    def render(self):
        # Return a string representation of the game state
        state_str = f"Current Player: Player {self.current_player} ({'Even' if self.assigned_parity[self.current_player] == 0 else 'Odd'})\n"
        state_str += f"Array: {self.array.tolist()}\n"
        return state_str

    def valid_moves(self):
        valid_actions = []
        for index in range(7):
            value = self.array[index]
            # Check increment action
            if value < 9:
                valid_actions.append(index * 2)  # Increment action
            # Check decrement action
            if value > 1:
                valid_actions.append(index * 2 + 1)  # Decrement action
        return valid_actions
