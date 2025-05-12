import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Initialize default target number
        self.default_target_number = 20

        # Define action space: 0 - Add 1, 1 - Multiply by 2
        self.action_space = spaces.Discrete(2)

        # Define observation space: [player's current number, opponent's current number, target number]
        max_number = np.iinfo(np.int32).max
        self.observation_space = spaces.Box(
            low=np.array([1, 1, 1], dtype=np.int32),
            high=np.array([max_number, max_number, max_number], dtype=np.int32),
            dtype=np.int32,
        )

        # Initialize state variables
        self.current_numbers = [1, 1]  # [Player 1 number, Player 2 number]
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.target_number = self.default_target_number
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Set the target number
        if options is not None and "target_number" in options:
            self.target_number = options["target_number"]
        else:
            self.target_number = self.default_target_number

        # Reset current numbers for both players
        self.current_numbers = [1, 1]

        # Determine starting player
        self.current_player = 0  # Player 1 starts; modify as needed

        # Reset game status
        self.done = False

        # Prepare observation
        observation = np.array(
            [
                self.current_numbers[self.current_player],  # Current player's number
                self.current_numbers[1 - self.current_player],  # Opponent's number
                self.target_number,
            ],
            dtype=np.int32,
        )

        return observation, {}  # Observation and info

    def step(self, action):
        if self.done:
            # Game has ended; return current state with no reward
            observation = np.array(
                [
                    self.current_numbers[self.current_player],
                    self.current_numbers[1 - self.current_player],
                    self.target_number,
                ],
                dtype=np.int32,
            )
            return observation, -10, True, False, {}

        # Apply the chosen operation
        current_number = self.current_numbers[self.current_player]
        if action == 0:
            # Add 1
            new_number = current_number + 1
        elif action == 1:
            # Multiply by 2
            new_number = current_number * 2
        else:
            # Invalid action
            observation = np.array(
                [
                    self.current_numbers[self.current_player],
                    self.current_numbers[1 - self.current_player],
                    self.target_number,
                ],
                dtype=np.int32,
            )
            return observation, -10, True, False, {}

        # Update the current number for the player
        self.current_numbers[self.current_player] = new_number

        # Check for victory or loss
        if new_number == self.target_number:
            # Current player wins
            self.done = True
            reward = 1
            terminated = True
        elif new_number > self.target_number:
            # Current player exceeds target and loses
            self.done = True
            reward = -10
            terminated = True
        else:
            # Game continues
            reward = 0
            terminated = False
            # Switch to the other player
            self.current_player = 1 - self.current_player

        # Prepare the observation for the next player
        observation = np.array(
            [
                self.current_numbers[self.current_player],
                self.current_numbers[1 - self.current_player],
                self.target_number,
            ],
            dtype=np.int32,
        )

        return observation, reward, terminated, False, {}

    def render(self):
        # Create a string representation of the current game state
        render_str = "--- Sequence Math Duel ---\n"
        render_str += f"Target Number: {self.target_number}\n"
        render_str += f"Player {self.current_player + 1}'s turn.\n"
        render_str += f"Player 1 Current Number: {self.current_numbers[0]}\n"
        render_str += f"Player 2 Current Number: {self.current_numbers[1]}\n"
        return render_str

    def valid_moves(self):
        # Both actions are always valid per game rules
        return [0, 1]
