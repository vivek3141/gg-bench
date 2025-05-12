import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: choose a number between 1 and 6 (actions 0 to 5)
        self.action_space = spaces.Discrete(6)

        # Observation space:
        # - Indices 0-19: numbers on array positions (values between 1 and 6)
        # - Indices 20-21: positions of Player 1 and Player 2 (values between 0 and 20)
        self.observation_space = spaces.Box(
            low=np.array([1] * 20 + [0, 0], dtype=np.int32),
            high=np.array([6] * 20 + [20, 20], dtype=np.int32),
            dtype=np.int32,
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the game array (positions 1 to 20 with random numbers between 1 and 6)
        self.array_numbers = np.random.randint(1, 7, size=20)

        # Initialize player positions (0 for both players at start)
        self.player_positions = {1: 0, 2: 0}

        # Randomly choose starting player
        self.current_player = np.random.choice([1, 2])

        # Build the observation
        self.observation = np.concatenate(
            (self.array_numbers, [self.player_positions[1], self.player_positions[2]])
        )

        self.done = False

        return self.observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.observation, 0, True, False, {}

        # Check if action is valid (0 to 5)
        if action < 0 or action >= 6:
            # Invalid action
            self.done = True
            return self.observation, -10, True, False, {"error": "Invalid action"}

        # Convert action to chosen number (1 to 6)
        chosen_number = action + 1

        # Get current player's position
        current_position = self.player_positions[self.current_player]

        # Find the next position ahead that contains the chosen number
        found = False
        for idx in range(current_position, 20):
            if self.array_numbers[idx] == chosen_number:
                next_position = idx + 1  # Positions are from 1 to 20
                found = True
                break

        if not found:
            # Invalid move (no matching number ahead)
            self.done = True
            reward = -10
            return self.observation, reward, True, False, {}

        # Valid move, update player's position
        self.player_positions[self.current_player] = next_position

        # Update observation
        self.observation[20 + self.current_player - 1] = next_position

        # Check for win condition
        if next_position >= 20:
            self.done = True
            reward = 1
            return self.observation, reward, True, False, {}
        else:
            # No reward for normal move
            reward = 0

        # Switch to next player
        self.current_player = 1 if self.current_player == 2 else 2

        return self.observation, reward, False, False, {}

    def render(self):
        # Build a string to represent the game state
        state_str = "Positions:\n"

        # Build the positions string
        for idx in range(20):
            pos_num = idx + 1
            number = self.array_numbers[idx]
            # Check if a player is on this position
            player_here = ""
            if self.player_positions[1] == pos_num:
                player_here += "P1 "
            if self.player_positions[2] == pos_num:
                player_here += "P2 "

            state_str += f"{pos_num:2d}:{number} {player_here}\n"

        state_str += f"Player 1 Position: {self.player_positions[1]}\n"
        state_str += f"Player 2 Position: {self.player_positions[2]}\n"
        state_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"

        return state_str

    def valid_moves(self):
        if self.done:
            return []

        current_position = self.player_positions[self.current_player]

        valid_actions = []
        for action in range(6):
            chosen_number = action + 1
            for idx in range(current_position, 20):
                if self.array_numbers[idx] == chosen_number:
                    valid_actions.append(action)
                    break  # No need to look further

        return valid_actions
