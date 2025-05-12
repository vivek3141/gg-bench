import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Six actions: Increment or decrement each of the three dials
        self.action_space = spaces.Discrete(6)

        # Observation space:
        # - Three dial values (0 to 9)
        # - Opponent's last dial adjusted (-1 for none, 0-2 for dials)
        # - Opponent's last move direction (-1 for decrement, 0 for none, 1 for increment)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, -1]),
            high=np.array([9, 9, 9, 2, 1]),
            dtype=np.int32,
        )

        # Initialize the lock and game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly initialize the lock combination
        self.dials = np.array([random.randint(0, 9) for _ in range(3)], dtype=np.int32)
        self.current_player = 0  # Player 0 starts
        self.opp_last_dial = -1  # No previous move
        self.opp_last_dir = 0  # No previous direction
        self.done = False
        # Prepare the initial observation
        observation = np.array(
            [*self.dials, self.opp_last_dial, self.opp_last_dir], dtype=np.int32
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            observation = np.array(
                [*self.dials, self.opp_last_dial, self.opp_last_dir], dtype=np.int32
            )
            return observation, 0, True, False, {}  # No reward, game terminated

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            observation = np.array(
                [*self.dials, self.opp_last_dial, self.opp_last_dir], dtype=np.int32
            )
            return observation, -10, True, False, {}

        # Map action to dial index and direction
        action_mapping = {
            0: (0, 1),  # Increment Dial 1
            1: (0, -1),  # Decrement Dial 1
            2: (1, 1),  # Increment Dial 2
            3: (1, -1),  # Decrement Dial 2
            4: (2, 1),  # Increment Dial 3
            5: (2, -1),  # Decrement Dial 3
        }
        dial_index, move_dir = action_mapping[action]

        # Adjust the selected dial
        self.dials[dial_index] = (self.dials[dial_index] + move_dir) % 10

        # Update opponent's last move for restrictions
        self.opp_last_dial = dial_index
        self.opp_last_dir = move_dir

        # Check for win condition
        if np.all(self.dials == 0):
            self.done = True
            observation = np.array(
                [*self.dials, self.opp_last_dial, self.opp_last_dir], dtype=np.int32
            )
            return observation, 1, True, False, {}

        # Switch to next player
        self.current_player = 1 - self.current_player

        # Prepare observation for the next player
        observation = np.array(
            [*self.dials, self.opp_last_dial, self.opp_last_dir], dtype=np.int32
        )
        return observation, 0, False, False, {}

    def render(self):
        # Return a string representation of the current state
        dial_state = f"{self.dials[0]}-{self.dials[1]}-{self.dials[2]}"
        return f"Current combination: {dial_state}"

    def valid_moves(self):
        # All possible actions
        valid_actions = list(range(6))

        # If there is a move restriction from opponent's last move
        if self.opp_last_dial != -1:
            action_mapping = {
                0: (0, 1),  # Increment Dial 1
                1: (0, -1),  # Decrement Dial 1
                2: (1, 1),  # Increment Dial 2
                3: (1, -1),  # Decrement Dial 2
                4: (2, 1),  # Increment Dial 3
                5: (2, -1),  # Decrement Dial 3
            }
            # Determine invalid action that reverses opponent's last move
            invalid_actions = []
            for action in valid_actions:
                dial_index, move_dir = action_mapping[action]
                if (dial_index == self.opp_last_dial) and (
                    move_dir == -self.opp_last_dir
                ):
                    invalid_actions.append(action)
            # Exclude invalid actions
            valid_actions = [a for a in valid_actions if a not in invalid_actions]
        return valid_actions
