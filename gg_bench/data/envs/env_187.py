import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to picking numbers from 1 to 10 (indices 0 to 9)
        self.action_space = spaces.Discrete(10)

        # Observation space:
        # - First 10 elements: 0 (available) or 1 (claimed)
        # - Last element: last number selected (0 if none)
        self.observation_space = spaces.Box(low=0, high=10, shape=(11,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.claimed_numbers = np.zeros(10, dtype=np.int8)  # Numbers 1 to 10
        self.last_number = 0  # 0 indicates no number has been selected yet
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = np.concatenate((self.claimed_numbers, [self.last_number]))
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is already over
            observation = np.concatenate((self.claimed_numbers, [self.last_number]))
            return observation, 0, True, False, {}

        if action < 0 or action >= 10:
            # Invalid action: out of bounds
            observation = np.concatenate((self.claimed_numbers, [self.last_number]))
            return observation, -10, True, False, {}

        number = action + 1  # Convert action to number (1 to 10)

        # Check if the number is already claimed
        if self.claimed_numbers[action] == 1:
            # Invalid move: number already claimed
            observation = np.concatenate((self.claimed_numbers, [self.last_number]))
            return observation, -10, True, False, {}

        # Check if the move is valid
        if self.last_number == 0:
            # First turn: any unclaimed number is valid
            valid_move = True
        else:
            # Check adjacency
            left_neighbor = (self.last_number - 2) % 10 + 1
            right_neighbor = self.last_number % 10 + 1
            if number == left_neighbor or number == right_neighbor:
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            # Invalid move: not adjacent to last number
            observation = np.concatenate((self.claimed_numbers, [self.last_number]))
            return observation, -10, True, False, {}

        # Valid move: update the game state
        self.claimed_numbers[action] = 1  # Mark the number as claimed
        self.last_number = number  # Update the last number selected

        # Check if the next player has any valid moves
        next_valid_moves = self.get_valid_moves()
        if not next_valid_moves:
            # Next player cannot make a move: current player wins
            self.done = True
            observation = np.concatenate((self.claimed_numbers, [self.last_number]))
            return observation, 1, True, False, {}

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1
        observation = np.concatenate((self.claimed_numbers, [self.last_number]))
        return observation, 0, False, False, {}  # Game continues

    def get_valid_moves(self):
        # Returns a list of valid moves (actions) for the current player
        if self.last_number == 0:
            # First turn: any unclaimed number
            return [i for i in range(10) if self.claimed_numbers[i] == 0]
        else:
            # Adjacent numbers to the last number selected
            left_neighbor = (self.last_number - 2) % 10 + 1
            right_neighbor = self.last_number % 10 + 1
            adj_indices = [left_neighbor - 1, right_neighbor - 1]
            return [idx for idx in adj_indices if self.claimed_numbers[idx] == 0]

    def valid_moves(self):
        # Return a list of valid moves (action indices)
        return self.get_valid_moves()

    def render(self):
        # Visual representation of the circle
        circle_str = "\nNumber Circle State:\n"
        for i in range(10):
            number = i + 1
            if self.claimed_numbers[i]:
                circle_str += f"({number}) "
            else:
                circle_str += f"[{number}] "
        circle_str += f"\nLast number selected: {self.last_number}"
        circle_str += f"\nCurrent player: Player {self.current_player}\n"
        return circle_str
