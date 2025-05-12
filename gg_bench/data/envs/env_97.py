import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            3
        )  # Actions: 0 - move 1 step, 1 - move 2 steps, 2 - move 3 steps
        self.observation_space = spaces.Box(low=0, high=4, shape=(11,), dtype=int)

        # Game constants
        self.START_POSITIONS = {1: 0, 2: 10}  # Player 1 starts at 0, Player 2 at 10
        self.OPPONENT_START = {1: 10, 2: 0}  # Opponent's starting position
        self.DIRECTION = {1: 1, 2: -1}  # Direction of movement for each player

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board
        self.board = np.zeros(11, dtype=int)

        # Initialize player positions
        self.player_positions = {1: self.START_POSITIONS[1], 2: self.START_POSITIONS[2]}

        # Randomly place traps for both players
        trap_positions = self.np_random.choice(range(1, 10), size=6, replace=False)
        self.player_traps = {
            1: trap_positions[:3].tolist(),
            2: trap_positions[3:].tolist(),
        }

        # Set own traps on the board for current player's observation
        for pos in self.player_traps[1]:
            self.board[pos] = 1  # Current player's traps are marked as 1

        # Initialize other state variables
        self.revealed_traps = set()
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Update player positions on the board
        self.board[self.player_positions[1]] = 3  # Player 1 marked as 3
        self.board[self.player_positions[2]] = 4  # Player 2 marked as 4

        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self.board.copy(), -10, True, False, {}  # Invalid move

        move_steps = (
            action + 1
        )  # Action 0 => move 1 step, Action 1 => move 2 steps, etc.

        # Remove current player position from the board
        if self.current_player == 1:
            self.board[self.player_positions[1]] = 0
        else:
            self.board[self.player_positions[2]] = 0

        for step in range(move_steps):
            # Compute new position
            new_position = (
                self.player_positions[self.current_player]
                + self.DIRECTION[self.current_player]
            )

            # Check if moving beyond the opponent's starting position
            if (self.current_player == 1 and new_position > 10) or (
                self.current_player == 2 and new_position < 0
            ):
                new_position = self.OPPONENT_START[self.current_player]

            # Check if new position is occupied by opponent
            if new_position == self.player_positions[3 - self.current_player]:
                # Cannot move onto opponent's position
                # End movement here
                self.player_positions[self.current_player] = self.player_positions[
                    self.current_player
                ]
                break
            else:
                self.player_positions[self.current_player] = new_position

            # Check for trap
            opponent_traps = self.player_traps[3 - self.current_player]
            if self.player_positions[self.current_player] in opponent_traps:
                # Trap triggered
                self.revealed_traps.add(self.player_positions[self.current_player])
                # Update the board to mark revealed trap
                self.board[self.player_positions[self.current_player]] = 2
                self.done = True
                reward = -1  # Current player loses
                return self.board.copy(), reward, True, False, {}

            # Check for victory
            if (
                self.player_positions[self.current_player]
                == self.OPPONENT_START[self.current_player]
            ):
                self.done = True
                reward = 1  # Current player wins
                if self.current_player == 1:
                    self.board[self.player_positions[1]] = 3
                else:
                    self.board[self.player_positions[2]] = 3
                return self.board.copy(), reward, True, False, {}

        # Update the board with new positions
        if self.current_player == 1:
            self.board[self.player_positions[1]] = 3  # Player 1 position
            self.board[self.player_positions[2]] = 4  # Player 2 position
        else:
            self.board[self.player_positions[2]] = 3  # Player 2 position
            self.board[self.player_positions[1]] = 4  # Player 1 position

        # Switch current player
        self.current_player = 3 - self.current_player

        # Update the board with own traps for the new current player
        self.board = self.update_board_for_player()

        # No reward for a standard move
        return self.board.copy(), 0, False, False, {}

    def update_board_for_player(self):
        # Reset board except for revealed traps and opponent's position
        new_board = np.zeros(11, dtype=int)

        # Own traps
        for pos in self.player_traps[self.current_player]:
            new_board[pos] = 1  # Current player's traps

        # Revealed traps
        for pos in self.revealed_traps:
            new_board[pos] = 2

        # Current player position
        new_board[self.player_positions[self.current_player]] = 3

        # Opponent's position
        new_board[self.player_positions[3 - self.current_player]] = 4

        return new_board

    def render(self):
        # Visual representation of the board
        symbols = {0: " . ", 1: " T ", 2: " X ", 3: " P ", 4: " O "}
        board_str = "Board:"
        for pos in range(11):
            board_str += symbols[self.board[pos]]
        return board_str

    def valid_moves(self):
        # Calculate valid moves for current player
        max_move = 3
        valid_actions = []
        for action in range(3):
            move_steps = action + 1
            temp_position = self.player_positions[self.current_player]
            valid = True
            for step in range(move_steps):
                new_position = temp_position + self.DIRECTION[self.current_player]
                # Check if moving beyond opponent's starting position
                if (self.current_player == 1 and new_position > 10) or (
                    self.current_player == 2 and new_position < 0
                ):
                    valid = False
                    break
                # Check if new position is occupied by opponent
                if new_position == self.player_positions[3 - self.current_player]:
                    valid = False
                    break
                temp_position = new_position
            if valid:
                valid_actions.append(action)
        return valid_actions
