import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the maximum length of the shared list
        self.max_length = 20  # Adjust as needed

        # Define action and observation spaces
        self.action_space = spaces.Discrete(18)  # 9 numbers * 2 positions (left/right)
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(self.max_length,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_list = []  # Initialize the shared list as empty
        self.current_player = 1  # Start with Player 1
        self.done = False  # Game is not over
        return self._get_obs(), {}  # Return observation and info

    def _get_obs(self):
        # Create the observation by padding the shared list to the maximum length
        obs = np.zeros(self.max_length, dtype=np.int32)
        obs[: len(self.shared_list)] = self.shared_list
        return obs

    def _satisfies_chain_rule(self, number1, number2):
        # Check if number1 is a factor or multiple of number2
        return number2 % number1 == 0 or number1 % number2 == 0

    def valid_moves(self):
        valid_actions = []
        for action in range(self.action_space.n):
            number = (action // 2) + 1  # Map action index to number (1-9)
            position = "left" if action % 2 == 0 else "right"  # Determine position

            if len(self.shared_list) == 0:
                # When the list is empty, any number can be added to either end
                valid_actions.append(action)
            else:
                if position == "left":
                    adjacent_number = self.shared_list[0]
                else:
                    adjacent_number = self.shared_list[-1]
                # Check if the Chain Rule is satisfied
                if self._satisfies_chain_rule(number, adjacent_number):
                    valid_actions.append(action)
        return valid_actions

    def step(self, action):
        if self.done:
            # If the game is over, return the current state
            return self._get_obs(), 0, True, False, {}

        valid_moves = self.valid_moves()
        if action not in valid_moves:
            # Invalid move: the player loses the game
            self.done = True
            reward = -10  # Penalty for invalid move
            return self._get_obs(), reward, True, False, {}

        # Map action index to number and position
        number = (action // 2) + 1  # Numbers from 1 to 9
        position = "left" if action % 2 == 0 else "right"

        # Add the number to the shared list at the specified end
        if position == "left":
            self.shared_list.insert(0, number)
        else:
            self.shared_list.append(number)

        # Switch to the next player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2

        # Check if the next player has any valid moves
        if not self.valid_moves():
            # Next player cannot move: current player wins
            self.done = True
            reward = 1  # Reward for winning
            return self._get_obs(), reward, True, False, {}

        # Game continues
        return self._get_obs(), 0, False, False, {}  # Reward is 0 for a valid move

    def render(self):
        # Return a string representation of the shared list
        return f"Shared List: {self.shared_list}"
