import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 6 possible actions: attack opponent's block at index 0-4, or action 5 ('pass' or 'attack fails')
        self.action_space = spaces.Discrete(6)  # Actions 0-5

        # Observation space: array of size 10, values from 0 to 5
        self.observation_space = spaces.Box(low=0, high=5, shape=(10,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the stacks
        self.state = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.state, {}  # Observation and info

    def step(self, action):
        if self.done:
            return self.state, 0, True, False, {}  # Game is already over

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            self.done = True
            reward = -10
            return self.state, reward, self.done, False, {}  # Invalid action

        # Determine player's and opponent's stack indices
        if self.current_player == 1:
            own_stack_indices = range(0, 5)
            opp_stack_indices = range(5, 10)
        else:
            own_stack_indices = range(5, 10)
            opp_stack_indices = range(0, 5)

        # Get attacking block (topmost block)
        attacking_block_index = None
        for i in reversed(own_stack_indices):
            if self.state[i] != 0:
                attacking_block_index = i
                attacking_block_value = self.state[i]
                break
        else:
            # Player has no blocks left
            self.done = True
            reward = -1  # Loss
            return self.state, reward, self.done, False, {}

        if action == 5:
            # Pass action (attack fails due to no valid targets)
            # Remove attacking block
            self.state[attacking_block_index] = 0
            # Check if player has any blocks left
            if all(self.state[own_stack_indices] == 0):
                # Player has no blocks left, loses the game
                self.done = True
                reward = -1  # Loss
                return self.state, reward, self.done, False, {}
            else:
                # Switch player
                self.current_player = 2 if self.current_player == 1 else 1
                return self.state, 0, self.done, False, {}
        else:
            # Valid attack action
            opp_block_index = opp_stack_indices[action]
            opp_block_value = self.state[opp_block_index]
            # Remove opponent's block
            self.state[opp_block_index] = 0
            # Remove attacking block
            self.state[attacking_block_index] = 0
            # Check if opponent has any blocks left
            if all(self.state[opp_stack_indices] == 0):
                # Opponent has no blocks left, current player wins
                self.done = True
                reward = 1  # Win
                return self.state, reward, self.done, False, {}
            else:
                # Switch player
                self.current_player = 2 if self.current_player == 1 else 1
                return self.state, 0, self.done, False, {}

    def render(self):
        # Return a visual representation of the environment state as a string
        s = ""
        s += f"Player 1's Stack: {self.state[0:5].tolist()}\n"
        s += f"Player 2's Stack: {self.state[5:10].tolist()}\n"
        s += f"It is Player {self.current_player}'s turn.\n"
        return s

    def valid_moves(self):
        if self.done:
            return []

        if self.current_player == 1:
            own_stack_indices = range(0, 5)
            opp_stack_indices = range(5, 10)
        else:
            own_stack_indices = range(5, 10)
            opp_stack_indices = range(0, 5)

        # Get attacking block (topmost block)
        attacking_block = None
        for i in reversed(own_stack_indices):
            if self.state[i] != 0:
                attacking_block = self.state[i]
                break
        else:
            # No blocks left
            return []

        # Find opponent's blocks that can be attacked
        valid_actions = []
        for idx, opp_idx in enumerate(opp_stack_indices):
            opp_block = self.state[opp_idx]
            if opp_block != 0 and opp_block <= attacking_block:
                valid_actions.append(idx)

        if not valid_actions:
            # No valid attacks, valid action is 5 (pass)
            return [5]
        else:
            return valid_actions
