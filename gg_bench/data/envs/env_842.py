import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action space:
        # 0: Subtract 1
        # 1: Subtract 2
        # 2: Multiply by 0.5 (if current number is even)
        # 3: Add 5 (if current number is less than 10)
        self.action_space = spaces.Discrete(4)

        # Observation space contains the numbers of both players
        # Player numbers range from 0 to 20
        self.observation_space = spaces.Box(
            low=0, high=20, shape=(2,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_numbers = [20, 20]  # Both players start with 20
        self.current_player = 0  # Player 1 starts
        self.done = False  # Game is not over
        return np.array(self.player_numbers, dtype=np.float32), {}

    def step(self, action):
        if self.done:
            raise Exception("Game is over. Please reset the environment.")

        current_number = self.player_numbers[self.current_player]

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid action selected
            self.done = True
            reward = -10  # Penalty for invalid move
            return (
                np.array(self.player_numbers, dtype=np.float32),
                reward,
                True,
                False,
                {},
            )

        # Apply the selected action
        new_number = self.apply_action(current_number, action)
        new_number = int(new_number)

        # Update the current player's number
        self.player_numbers[self.current_player] = new_number

        # Check for win condition
        if new_number == 0:
            self.done = True
            reward = 1  # Current player wins
            return (
                np.array(self.player_numbers, dtype=np.float32),
                reward,
                True,
                False,
                {},
            )

        # Check for loss condition
        if new_number < 0:
            self.done = True
            reward = -1  # Current player loses
            return (
                np.array(self.player_numbers, dtype=np.float32),
                reward,
                True,
                False,
                {},
            )

        # Continue the game
        self.current_player = 1 - self.current_player  # Switch players
        reward = 0  # No reward for regular move
        return np.array(self.player_numbers, dtype=np.float32), reward, False, False, {}

    def render(self):
        state_str = f"Player 1's number: {self.player_numbers[0]}\n"
        state_str += f"Player 2's number: {self.player_numbers[1]}\n"
        state_str += (
            f"Current player: {'Player 1' if self.current_player == 0 else 'Player 2'}"
        )
        print(state_str)

    def valid_moves(self):
        current_number = self.player_numbers[self.current_player]
        allowed_actions = []

        # Always available actions
        allowed_actions.append(0)  # Subtract 1
        allowed_actions.append(1)  # Subtract 2

        # Multiply by 0.5 if number is even
        if current_number % 2 == 0:
            allowed_actions.append(2)

        # Add 5 if number is less than 10
        if current_number < 10:
            allowed_actions.append(3)

        # Filter actions that keep number zero or positive
        valid_actions = []
        for action in allowed_actions:
            new_number = self.apply_action(current_number, action)
            if new_number >= 0:
                valid_actions.append(action)

        if valid_actions:
            return valid_actions  # Actions that keep number zero or positive
        else:
            return allowed_actions  # Must choose an action leading to negative number

    def apply_action(self, current_number, action):
        if action == 0:
            return current_number - 1
        elif action == 1:
            return current_number - 2
        elif action == 2:
            return current_number * 0.5
        elif action == 3:
            return current_number + 5
        else:
            raise ValueError("Invalid action.")
