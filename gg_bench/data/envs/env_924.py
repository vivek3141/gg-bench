import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: integers from 0 to 8 corresponding to numbers 1 to 9
        self.action_space = spaces.Discrete(9)
        # Observation space: current total sum, integer from 0 to 50
        self.observation_space = spaces.Box(low=0, high=50, shape=(1,), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_total = 0
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.stack = []
        return (
            np.array([self.current_total], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            return np.array([self.current_total], dtype=np.int32), 0, True, False, {}

        if not self.action_space.contains(action):
            # Invalid action (not in action space)
            self.done = True
            return (
                np.array([self.current_total], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        chosen_number = action + 1  # Map action 0-8 to number 1-9

        if chosen_number + self.current_total > 50:
            # Invalid move (total sum exceeds 50)
            self.done = True
            return (
                np.array([self.current_total], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )
        else:
            self.stack.append(chosen_number)
            self.current_total += chosen_number

            if self.current_total == 50:
                # Current player wins by reaching total sum of 50
                self.done = True
                return (
                    np.array([self.current_total], dtype=np.int32),
                    1,
                    True,
                    False,
                    {},
                )
            else:
                # Check if opponent has any valid moves
                max_valid_move = min(9, 50 - self.current_total)

                if max_valid_move < 1:
                    # Opponent cannot make a valid move, current player wins
                    self.done = True
                    return (
                        np.array([self.current_total], dtype=np.int32),
                        1,
                        True,
                        False,
                        {},
                    )
                else:
                    # Game continues, switch to next player
                    self.current_player *= -1
                    return (
                        np.array([self.current_total], dtype=np.int32),
                        0,
                        False,
                        False,
                        {},
                    )

    def render(self):
        # Return a string representation of the current state
        state_str = f"Current Total Sum: {self.current_total}\n"
        state_str += f"Stack: {self.stack}\n"
        return state_str

    def valid_moves(self):
        # Return a list of valid actions based on the current total sum
        max_valid_number = min(9, 50 - self.current_total)
        valid_actions = [i for i in range(9) if (i + 1) <= max_valid_number]
        return valid_actions
