import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to numbers from 1 to 9
        self.action_space = spaces.Discrete(9)
        # Observation space contains [current_sum, last_number_placed]
        self.observation_space = spaces.Box(
            low=np.array([0, 1]), high=np.array([50, 9]), shape=(2,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_sum = 0
        self.last_number = 0  # No number has been placed yet
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.stack = []  # Initialize an empty stack

        # Observation is [current_sum, last_number_placed]
        observation = np.array([self.current_sum, self.last_number], dtype=np.float32)
        return observation, {}  # Return observation and info

    def step(self, action):
        # Convert action to the actual number (1-9)
        chosen_number = action + 1

        # Check if game is already over
        if self.done:
            return (
                np.array([self.current_sum, self.last_number], dtype=np.float32),
                -10,
                True,
                False,
                {},
            )

        # Check if the action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            observation = np.array(
                [self.current_sum, self.last_number], dtype=np.float32
            )
            return observation, reward, True, False, {}

        # Update the game state
        self.stack.append(chosen_number)
        self.current_sum += chosen_number
        self.last_number = chosen_number

        # Check for win condition
        if self.current_sum == 50:
            # Current player wins
            self.done = True
            reward = 1
            observation = np.array(
                [self.current_sum, self.last_number], dtype=np.float32
            )
            return observation, reward, True, False, {}
        elif self.current_sum > 50:
            # Current player loses
            self.done = True
            reward = -10
            observation = np.array(
                [self.current_sum, self.last_number], dtype=np.float32
            )
            return observation, reward, True, False, {}

        # Check if next player has valid moves
        self.current_player *= -1  # Switch player
        next_valid_actions = self.valid_moves()
        if not next_valid_actions:
            # Next player cannot make a valid move
            self.done = True
            # Current player wins because opponent cannot move
            self.current_player *= -1  # Switch back to current player
            reward = 1
            observation = np.array(
                [self.current_sum, self.last_number], dtype=np.float32
            )
            return observation, reward, True, False, {}

        # Game continues
        reward = 0
        observation = np.array([self.current_sum, self.last_number], dtype=np.float32)
        return observation, reward, False, False, {}

    def render(self):
        # Return a string representation of the stack and the sum
        stack_str = "Stack: " + str(self.stack)
        sum_str = "Total Sum: " + str(self.current_sum)
        return stack_str + "\n" + sum_str

    def valid_moves(self):
        # Valid actions are those that meet the game's criteria
        valid_actions = []

        # Possible numbers are from last_number to 9 inclusive
        min_number = max(self.last_number, 1)
        for action in range(9):
            number = action + 1
            if number >= min_number:
                if self.current_sum + number <= 50:
                    valid_actions.append(action)

        return valid_actions
