import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: integers from 2 to 9 inclusive (mapped from action indices 0 to 7)
        self.action_space = spaces.Discrete(8)

        # Observation space: [current_total, current_player]
        # current_total ranges from 1 upwards (set high to 1000 to accommodate totals exceeding 100)
        # current_player is either +1 (Player 1) or -1 (Player 2)
        self.observation_space = spaces.Box(
            low=np.array([1, -1]), high=np.array([1000, 1]), shape=(2,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_total = 1
        self.current_player = 1  # +1 for Player 1, -1 for Player 2
        self.done = False
        return np.array([self.current_total, self.current_player]), {}

    def step(self, action):
        if self.done:
            # Game is already over
            return (
                np.array([self.current_total, self.current_player]),
                -10,
                True,
                False,
                {},
            )

        selected_number = action + 2  # Map action index to number between 2 and 9

        # Multiply current total by selected number
        new_total = self.current_total * selected_number

        # Update the cumulative total and check for win or loss
        if new_total == 100:
            # Current player wins
            self.current_total = new_total
            self.done = True
            reward = 1  # Reward for winning
        elif new_total > 100:
            # Current player loses
            self.current_total = new_total
            self.done = True
            reward = -10  # Penalty for losing
        else:
            # Game continues
            self.current_total = new_total
            # Switch to the other player
            self.current_player *= -1
            reward = -10  # Penalty for a valid move

        return (
            np.array([self.current_total, self.current_player]),
            reward,
            self.done,
            False,
            {},
        )

    def render(self):
        state_str = f"Current Total: {self.current_total}\n"
        state_str += (
            f"{'Player 1' if self.current_player == 1 else 'Player 2'}'s Turn\n"
        )
        return state_str

    def valid_moves(self):
        # Return a list of valid action indices based on the current total
        valid_actions = []
        for action in range(8):
            selected_number = action + 2
            if self.current_total * selected_number <= 100:
                valid_actions.append(action)
        # If no moves can keep the total at or below 100, all moves are valid (player will lose)
        if not valid_actions:
            valid_actions = list(range(8))
        return valid_actions
