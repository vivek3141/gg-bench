import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0-3 for adding numbers 1-4, 4 for Attack
        self.action_space = spaces.Discrete(5)

        # Define observation space:
        # [current player's stack total, current player's top number,
        #  opponent's stack total, opponent's top number]
        self.observation_space = spaces.Box(low=0, high=10, shape=(4,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1  # Player 1 starts
        self.stacks = {1: [], -1: []}  # Stacks for Player 1 and Player 2
        self.totals = {1: 0, -1: 0}  # Totals for Player 1 and Player 2
        self.done = False

        return self._get_obs(), {}  # Observation, info

    def _get_obs(self):
        # Observation consists of:
        # [current player's stack total, current player's top number (0 if empty),
        #  opponent's stack total, opponent's top number (0 if empty)]
        cp = self.current_player
        op = -self.current_player

        cp_stack = self.stacks[cp]
        op_stack = self.stacks[op]

        cp_total = self.totals[cp]
        op_total = self.totals[op]

        cp_top = cp_stack[-1] if cp_stack else 0
        op_top = op_stack[-1] if op_stack else 0

        return np.array([cp_total, cp_top, op_total, op_top], dtype=np.int32)

    def valid_moves(self):
        valid_actions = []

        cp = self.current_player
        op = -self.current_player
        cp_stack = self.stacks[cp]
        op_stack = self.stacks[op]
        cp_total = self.totals[cp]

        # Add number actions (0-3)
        for action in range(4):
            number_to_add = action + 1  # Numbers 1-4
            if cp_total + number_to_add <= 10:
                valid_actions.append(action)

        # Attack action (action 4)
        if cp_stack and op_stack and cp_stack[-1] == op_stack[-1]:
            valid_actions.append(4)

        return valid_actions

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        cp = self.current_player
        op = -self.current_player
        cp_stack = self.stacks[cp]
        op_stack = self.stacks[op]
        cp_total = self.totals[cp]
        op_total = self.totals[op]

        reward = 0
        terminated = False
        truncated = False

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid action
            reward = -10
            terminated = True
            self.done = True
            return self._get_obs(), reward, terminated, truncated, {}

        if action in [0, 1, 2, 3]:  # Add Number
            number_to_add = action + 1  # Numbers 1-4

            cp_total += number_to_add
            cp_stack.append(number_to_add)
            self.totals[cp] = cp_total

            if cp_total == 10:
                # Current player wins
                reward = 1
                terminated = True
                self.done = True
                return self._get_obs(), reward, terminated, truncated, {}

            elif cp_total > 10:
                # Current player loses
                reward = -10
                terminated = True
                self.done = True
                return self._get_obs(), reward, terminated, truncated, {}
            else:
                # Game continues
                reward = 0

        elif action == 4:  # Attack
            # Remove top number from opponent's stack
            removed_number = op_stack.pop()
            op_total -= removed_number
            self.totals[op] = op_total

            # No need to check for opponent's bust as their total decreases

        # Switch to next player
        self.current_player *= -1

        # Return the observation for the new current player
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        cp = self.current_player
        op = -self.current_player
        cp_stack = self.stacks[cp]
        op_stack = self.stacks[op]

        cp_total = self.totals[cp]
        op_total = self.totals[op]

        s = f"Player {cp}'s turn\n"
        s += "------------------------\n"
        s += f"Player {cp} Stack: {cp_stack}, Total: {cp_total}\n"
        s += f"Player {op} Stack: {op_stack}, Total: {op_total}\n"
        s += "------------------------\n"
        return s

    def valid_moves(self):
        valid_actions = []

        cp = self.current_player
        op = -self.current_player
        cp_stack = self.stacks[cp]
        op_stack = self.stacks[op]
        cp_total = self.totals[cp]

        # Add number actions (0-3)
        for action in range(4):
            number_to_add = action + 1  # Numbers 1-4
            if cp_total + number_to_add <= 10:
                valid_actions.append(action)

        # Attack action (action 4)
        if cp_stack and op_stack and cp_stack[-1] == op_stack[-1]:
            valid_actions.append(4)

        return valid_actions
