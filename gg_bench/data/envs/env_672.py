import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces
        # Actions are integers from 0 to 8, corresponding to move numbers from 1 to 9
        self.action_space = spaces.Discrete(9)

        # Observations are the positions of Player 1 and Player 2 on the staircase (steps 0 to 20)
        self.observation_space = spaces.Box(low=0, high=20, shape=(2,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Both players start at step 0
        self.positions = [0, 0]  # positions[0] is Player 1, positions[1] is Player 2
        self.current_player = 1  # Player 1 starts the game
        self.done = False  # Game over flag
        return np.array(self.positions, dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # If the game is over, no more moves are allowed
            return np.array(self.positions, dtype=np.int32), 0, True, False, {}

        # Get the list of valid moves for the current player
        valid_moves_list = self.valid_moves()

        # If the current player has no valid moves, they must pass their turn
        if len(valid_moves_list) == 0:
            # Switch to the next player
            self.current_player = 3 - self.current_player
            return np.array(self.positions, dtype=np.int32), 0, False, False, {}

        if action not in valid_moves_list:
            # Invalid move ends the game with a penalty
            self.done = True
            return np.array(self.positions, dtype=np.int32), -10, True, False, {}

        # Map action to move number (1-9)
        move_number = action + 1

        # Get indices for the current player and opponent
        player_idx = self.current_player - 1
        opponent_idx = 1 - player_idx

        # Calculate new position
        current_pos = self.positions[player_idx]
        new_pos = current_pos + move_number

        # Update player's position
        self.positions[player_idx] = new_pos

        # Check for victory condition
        if new_pos == 20:
            self.done = True
            return np.array(self.positions, dtype=np.int32), 1, True, False, {}

        # Check for prime number bonus
        primes = {2, 3, 5, 7, 11, 13, 17, 19}
        if new_pos in primes:
            # Extra turn granted; do not switch players
            return np.array(self.positions, dtype=np.int32), 0, False, False, {}
        else:
            # Switch players
            self.current_player = 3 - self.current_player
            return np.array(self.positions, dtype=np.int32), 0, False, False, {}

    def render(self):
        # Return a visual representation of the game state as a string
        output = "--- Prime Climb Game State ---\n"
        for step in range(20, 0, -1):
            line = "Step {:>2}: ".format(step)
            if self.positions[0] == step and self.positions[1] == step:
                line += "P1 & P2"
            elif self.positions[0] == step:
                line += "P1"
            elif self.positions[1] == step:
                line += "P2"
            else:
                line += ""
            output += line + "\n"
        output += "Off the staircase (Step 0): "
        if self.positions[0] == 0 and self.positions[1] == 0:
            output += "P1 & P2"
        elif self.positions[0] == 0:
            output += "P1"
        elif self.positions[1] == 0:
            output += "P2"
        else:
            output += ""
        output += "\n------------------------------\n"
        output += "Current Player: P{}\n".format(self.current_player)
        return output

    def valid_moves(self):
        # Returns a list of valid moves (action indices) for the current player
        valid_actions = []

        # Map action indices to move numbers (1 to 9)
        possible_actions = range(9)  # Actions 0 to 8

        # Get indices for the current player and opponent
        player_idx = self.current_player - 1
        opponent_idx = 1 - player_idx

        current_pos = self.positions[player_idx]
        opponent_pos = self.positions[opponent_idx]

        for action in possible_actions:
            move_number = action + 1
            new_pos = current_pos + move_number

            # Check if the move is within the staircase limits
            if new_pos > 20:
                continue  # Move exceeds Step 20; invalid

            # Check if the new position is occupied by the opponent
            if new_pos == opponent_pos:
                continue  # Move lands on opponent's step; invalid

            # Move is valid
            valid_actions.append(action)

        return valid_actions
