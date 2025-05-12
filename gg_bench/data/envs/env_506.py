import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=4):
        super(CustomEnv, self).__init__()

        self.N = N  # Size of the array
        self.num_swaps = N - 1
        self.num_rotations = (N - 2) * 2  # Each position can rotate left or right
        self.total_actions = self.num_swaps + self.num_rotations

        # Define action and observation space
        self.action_space = spaces.Discrete(self.total_actions)
        self.observation_space = spaces.Box(
            low=1, high=self.N, shape=(self.N,), dtype=np.int32
        )

        # Generate action mappings
        self.action_to_move = {}
        action_idx = 0
        # Swaps
        for i in range(self.N - 1):
            self.action_to_move[action_idx] = ("swap", i)
            action_idx += 1
        # Rotations
        for i in range(self.N - 2):
            self.action_to_move[action_idx] = ("rotate", i, "L")
            action_idx += 1
            self.action_to_move[action_idx] = ("rotate", i, "R")
            action_idx += 1

        # Inverse mapping from move to action index
        self.move_to_action = {v: k for k, v in self.action_to_move.items()}

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the array with values from 1 to N in random order
        self.array = np.random.permutation(np.arange(1, self.N + 1))
        self.current_player = 1  # Player 1 starts
        self.last_move = None  # No previous move
        self.done = False
        return self.array.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.array.copy(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            reward = -10
            info = {"invalid_move": True}
            # Turn passes to the opponent
            self.current_player *= -1
            return self.array.copy(), reward, False, False, info

        # Apply the action
        move = self.action_to_move[action]
        self.apply_move(move)
        # Update last_move to prevent opponent from undoing
        self.last_move = move

        # Check for win condition
        if np.all(self.array == np.sort(self.array)):
            # Current player wins
            reward = 1
            self.done = True
            return self.array.copy(), reward, True, False, {}
        else:
            reward = 0

        # Turn passes to opponent
        self.current_player *= -1

        return self.array.copy(), reward, False, False, {}

    def apply_move(self, move):
        if move[0] == "swap":
            _, idx = move
            # Swap elements at idx and idx + 1
            self.array[idx], self.array[idx + 1] = self.array[idx + 1], self.array[idx]
        elif move[0] == "rotate":
            _, idx, direction = move
            if direction == "L":
                # Left rotation
                temp = self.array[idx]
                self.array[idx : idx + 2] = self.array[idx + 1 : idx + 3]
                self.array[idx + 2] = temp
            elif direction == "R":
                # Right rotation
                temp = self.array[idx + 2]
                self.array[idx + 1 : idx + 3] = self.array[idx : idx + 2]
                self.array[idx] = temp

    def valid_moves(self):
        valid_actions = list(range(self.total_actions))
        # Exclude the inverse of opponent's last move
        if self.last_move is not None:
            inverse_move = self.get_inverse_move(self.last_move)
            inverse_action = None
            # Find the action index of the inverse move
            for action_idx, move in self.action_to_move.items():
                if move == inverse_move:
                    inverse_action = action_idx
                    break
            if inverse_action is not None and inverse_action in valid_actions:
                valid_actions.remove(inverse_action)
        return valid_actions

    def get_inverse_move(self, move):
        if move[0] == "swap":
            # Inverse of swap is swap at the same position
            return move
        elif move[0] == "rotate":
            # Inverse of rotate is rotate at same position in opposite direction
            direction_opposite = "R" if move[2] == "L" else "L"
            return ("rotate", move[1], direction_opposite)

    def render(self):
        board_str = f"Current array: {self.array.tolist()}\n"
        board_str += f"Player {self.current_player}'s turn."
        return board_str

    def valid_moves_list(self):
        # Returns the list of valid move descriptions.
        moves = []
        for action in self.valid_moves():
            move = self.action_to_move[action]
            if move[0] == "swap":
                moves.append(f"Swap at position {move[1]}")
            elif move[0] == "rotate":
                moves.append(f"Rotate at position {move[1]} direction {move[2]}")
        return moves

    def close(self):
        pass
