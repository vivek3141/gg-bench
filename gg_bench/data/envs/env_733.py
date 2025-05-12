import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to integers from 0 to 100 (the numbers to subtract)
        self.action_space = spaces.Discrete(101)

        # Observation is the current number, ranging from 0 to 100
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 100
        self.current_player = 1  # Player 1 starts; Player 1: 1, Player 2: -1
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}  # Observation, info

    def step(self, action):
        if self.done:
            # Game has ended
            return (
                np.array([self.current_number], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Valid action; subtract the action from the current number
        self.current_number -= action

        if self.current_number == 0:
            # Current player wins by reducing number to 0
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                1,
                True,
                False,
                {},
            )
        else:
            # Check if the next player has any valid moves
            if len(self.get_proper_divisors(self.current_number)) == 0:
                # Current player wins; opponent has no valid moves
                self.done = True
                return (
                    np.array([self.current_number], dtype=np.int32),
                    1,
                    True,
                    False,
                    {},
                )
            else:
                # Switch to the other player
                self.current_player *= -1
                return (
                    np.array([self.current_number], dtype=np.int32),
                    0,
                    False,
                    False,
                    {},
                )

    def render(self):
        render_str = "--- Divisor Duel ---\n"
        render_str += f"Player {1 if self.current_player == 1 else 2}'s turn.\n"
        render_str += f"Current Number: {self.current_number}\n"
        valid_actions = self.valid_moves()
        if valid_actions:
            render_str += f"Available Divisors: {', '.join(map(str, valid_actions))}\n"
        else:
            render_str += "No valid moves available.\n"
        return render_str

    def valid_moves(self):
        # Return a list of valid actions (proper divisors of the current number)
        return self.get_proper_divisors(self.current_number)

    def get_proper_divisors(self, number):
        # Proper divisors are numbers greater than 1 and less than the number itself
        divisors = []
        for i in range(2, number):
            if number % i == 0:
                divisors.append(i)
        return divisors
