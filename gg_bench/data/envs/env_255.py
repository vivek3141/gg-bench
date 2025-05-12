import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 9 possible actions (numbers 1-9)
        self.action_space = spaces.Discrete(9)

        # Observation space consists of:
        # - Remaining numbers (indices 0-8): 1 if available, 0 if used
        # - Stack (indices 9-17): numbers placed so far (0 if empty)
        # - Current pattern (index 18): 1 for '>', -1 for '<', 0 for N/A
        self.observation_space = spaces.Box(low=-1, high=9, shape=(19,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.remaining_numbers = np.ones(9, dtype=np.int32)  # Numbers 1-9 are available
        self.stack = np.zeros(9, dtype=np.int32)  # Stack of numbers played
        self.stack_top_index = -1  # Indicates the top of the stack (-1 means empty)
        self.current_pattern = None  # No pattern for the first move
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.info = {}
        observation = self._get_observation()
        return observation, self.info  # Return observation and info

    def _get_observation(self):
        obs = np.zeros(19, dtype=np.int32)
        obs[0:9] = self.remaining_numbers
        obs[9:18] = self.stack
        if self.current_pattern == ">":
            obs[18] = 1
        elif self.current_pattern == "<":
            obs[18] = -1
        else:
            obs[18] = 0  # No pattern for first move
        return obs

    def step(self, action):
        if self.done:
            # Game is over, cannot take any more actions
            observation = self._get_observation()
            return (
                observation,
                0,
                True,
                False,
                self.info,
            )  # Observation, reward, terminated, truncated, info

        # Check if current player has any valid moves
        valid_actions = self.valid_moves()
        if not valid_actions:
            # Current player cannot move, previous player wins
            self.done = True
            observation = self._get_observation()
            reward = -1  # Current player loses
            return observation, reward, True, False, self.info

        # Check if action is valid
        if action not in valid_actions:
            # Invalid action
            self.done = True
            observation = self._get_observation()
            reward = -10  # Invalid move
            return observation, reward, True, False, self.info

        # Map action to number (action 0 corresponds to number 1)
        number = action + 1

        # Perform the move
        self.remaining_numbers[action] = 0  # Remove the number from remaining_numbers
        self.stack_top_index += 1
        self.stack[self.stack_top_index] = number  # Place the number on the stack

        # Switch pattern
        if self.stack_top_index == 0:
            # First move, set pattern to '>'
            self.current_pattern = ">"
        else:
            # Switch pattern
            if self.current_pattern == ">":
                self.current_pattern = "<"
            else:
                self.current_pattern = ">"

        # Switch current player
        self.current_player *= -1

        # Check if the next player (new current_player) has any valid moves
        valid_actions = self.valid_moves()
        if not valid_actions:
            # New current player cannot move, the player who just moved wins
            self.done = True
            observation = self._get_observation()
            reward = +1  # The player who just moved wins
            return observation, reward, True, False, self.info

        # Otherwise, continue game
        observation = self._get_observation()
        reward = -10  # Valid move
        return observation, reward, False, False, self.info

    def render(self):
        state_str = "\nAvailable Numbers: "
        state_str += ", ".join(
            [str(i + 1) for i in range(9) if self.remaining_numbers[i] == 1]
        )
        state_str += "\nStack: "
        state_str += ", ".join(
            [str(int(n)) for n in self.stack[: self.stack_top_index + 1]]
        )
        state_str += "\nCurrent Pattern: "
        state_str += self.current_pattern if self.current_pattern else "N/A"
        state_str += "\nCurrent Player: "
        state_str += "Player 1" if self.current_player == 1 else "Player 2"
        return state_str

    def valid_moves(self):
        # Returns list of valid action indices for the current player
        valid_actions = []
        for action in range(9):
            if self.remaining_numbers[action] == 1:
                number = action + 1
                if self.stack_top_index >= 0:
                    top_number = self.stack[self.stack_top_index]
                    if self.current_pattern == ">":
                        if number > top_number:
                            valid_actions.append(action)
                    elif self.current_pattern == "<":
                        if number < top_number:
                            valid_actions.append(action)
                else:
                    # First move, any number is valid
                    valid_actions.append(action)
        return valid_actions
