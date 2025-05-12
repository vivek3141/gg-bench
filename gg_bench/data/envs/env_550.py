import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 20 possible actions
        # Actions 0-9: Decrement Counter A by amounts 1-10
        # Actions 10-19: Decrement Counter B by amounts 1-10
        self.action_space = spaces.Discrete(20)

        # Observation space: Two counters with values from 0 to 10
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([10, 10]), shape=(2,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # Initialize the counters
        self.counter_A = 10
        self.counter_B = 10

        # Player 1 starts (+1 for Player 1, -1 for Player 2)
        self.current_player = 1

        # Game status
        self.done = False

        # Return the initial observation and info
        return np.array([self.counter_A, self.counter_B]), {}

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return (
                np.array([self.counter_A, self.counter_B]),
                -10,
                True,
                False,
                {},
            )

        # Map the action to counter and decrement amount
        if 0 <= action <= 9:
            # Actions 0-9 correspond to Counter A decrements
            counter = "A"
            decrement = action + 1  # Decrement amounts from 1 to 10
        elif 10 <= action <= 19:
            # Actions 10-19 correspond to Counter B decrements
            counter = "B"
            decrement = action - 9  # Decrement amounts from 1 to 10
        else:
            # Invalid action
            self.done = True
            return (
                np.array([self.counter_A, self.counter_B]),
                -10,
                True,
                False,
                {},
            )

        # Perform the decrement if valid
        if counter == "A":
            if 1 <= decrement <= self.counter_A:
                self.counter_A -= decrement
            else:
                # Invalid move
                self.done = True
                return (
                    np.array([self.counter_A, self.counter_B]),
                    -10,
                    True,
                    False,
                    {},
                )
        elif counter == "B":
            if 1 <= decrement <= self.counter_B:
                self.counter_B -= decrement
            else:
                # Invalid move
                self.done = True
                return (
                    np.array([self.counter_A, self.counter_B]),
                    -10,
                    True,
                    False,
                    {},
                )

        # Check for the victory condition
        if self.counter_A == 0 and self.counter_B == 0:
            # Current player wins
            self.done = True
            return (
                np.array([self.counter_A, self.counter_B]),
                1,  # Reward for winning
                True,
                False,
                {},
            )

        # Switch to the next player
        self.current_player *= -1

        # Return the new observation
        return (
            np.array([self.counter_A, self.counter_B]),
            0,
            False,
            False,
            {},
        )

    def render(self):
        # Return a string representation of the game state
        output = f"Counter A: {self.counter_A}, Counter B: {self.counter_B}\n"
        output += (
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"
        )
        return output

    def valid_moves(self):
        # Generate a list of valid action indices based on the current counter values
        valid_actions = []

        # Valid actions for Counter A
        if self.counter_A > 0:
            for i in range(self.counter_A):
                action_idx = i  # Actions 0 to (counter_A - 1)
                valid_actions.append(action_idx)

        # Valid actions for Counter B
        if self.counter_B > 0:
            for i in range(self.counter_B):
                action_idx = 10 + i  # Actions 10 to (10 + counter_B - 1)
                valid_actions.append(action_idx)

        return valid_actions
