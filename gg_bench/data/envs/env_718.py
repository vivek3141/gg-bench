import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces
        # Actions: 0 = move forward 1 position
        #          1 = move forward 2 positions
        #          2 = move forward 3 positions
        #          3 = skip turn
        self.action_space = spaces.Discrete(4)

        # Observation space: 9 positions on the track
        # Values: 0 = empty, 1 = current player's marker, -1 = opponent's marker
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the board
        self.board = np.zeros(9, dtype=np.int8)

        # Set initial positions
        self.board[0] = 1  # Player 1 starts at position 1 (index 0)
        self.board[8] = -1  # Player 2 starts at position 9 (index 8)

        # Store player positions
        self.player_positions = {1: 0, -1: 8}

        # Starting positions
        self.starting_positions = {1: 0, -1: 8}

        # Set current player (1 or -1)
        self.current_player = 1

        # Game over flag
        self.done = False

        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        # Check if action is valid
        if action not in [0, 1, 2, 3]:
            return self.board.copy(), -10, True, False, {}

        if self.done:
            return self.board.copy(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            return self.board.copy(), -10, True, False, {}

        reward = 0

        if action == 3:
            # Skip turn
            self.current_player *= -1
            return self.board.copy(), reward, False, False, {}

        # Map action to move distance (1, 2, or 3)
        move_distance = action + 1

        # Current position
        current_pos = self.player_positions[self.current_player]

        # Calculate new position
        direction = 1 if self.current_player == 1 else -1
        new_pos = current_pos + move_distance * direction

        # Opponent's starting position
        opponent_start_pos = self.starting_positions[-self.current_player]

        # Check for win condition
        if (self.current_player == 1 and new_pos > opponent_start_pos) or (
            self.current_player == -1 and new_pos < opponent_start_pos
        ):
            self.done = True
            reward = 1
            return self.board.copy(), reward, True, False, {}

        # Check if new position is within bounds
        if new_pos < 0 or new_pos > 8:
            # Invalid move
            return self.board.copy(), -10, True, False, {}

        # Check if new position is occupied by opponent
        opponent_pos = self.player_positions[-self.current_player]
        if new_pos == opponent_pos:
            # Invalid move
            return self.board.copy(), -10, True, False, {}

        # Update positions
        self.board[current_pos] = 0
        self.board[new_pos] = self.current_player
        self.player_positions[self.current_player] = new_pos

        # Switch players
        self.current_player *= -1

        return self.board.copy(), reward, False, False, {}

    def render(self):
        # Generate a visual representation of the track
        track_positions = "Positions: " + " ".join(str(i + 1) for i in range(9)) + "\n"
        markers = "Markers:   "

        for i in range(9):
            if self.board[i] == 1:
                markers += "P1 "
            elif self.board[i] == -1:
                markers += "P2 "
            else:
                markers += "   "

        return track_positions + markers

    def valid_moves(self):
        # If the game is over, no valid moves
        if self.done:
            return []

        possible_actions = []
        current_pos = self.player_positions[self.current_player]
        opponent_pos = self.player_positions[-self.current_player]
        opponent_start_pos = self.starting_positions[-self.current_player]

        for action in [0, 1, 2]:
            move_distance = action + 1
            direction = 1 if self.current_player == 1 else -1
            new_pos = current_pos + move_distance * direction

            # Check for movements that result in a win
            if (self.current_player == 1 and new_pos > opponent_start_pos) or (
                self.current_player == -1 and new_pos < opponent_start_pos
            ):
                possible_actions.append(action)
                continue

            # Check if new position is within bounds
            if new_pos < 0 or new_pos > 8:
                continue  # Out of bounds, invalid move

            # Check if new position is occupied by opponent
            if new_pos == opponent_pos:
                continue  # Can't move to opponent's position

            possible_actions.append(action)

        if not possible_actions:
            possible_actions.append(3)  # Must skip turn

        return possible_actions
