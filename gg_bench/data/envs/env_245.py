import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):

    def __init__(self):

        super(CustomEnv, self).__init__()

        # Define action space
        # There are 18 possible actions, combining move (1 or 2 spaces) and blockade positions (1 to 9)
        self.action_space = spaces.Discrete(18)

        # Define observation space
        # Observation is an array of 11 positions, each can be 0 (empty), 1 (Player 1), 2 (Player 2), or 3 (blockade)
        self.observation_space = spaces.Box(low=0, high=3, shape=(11,), dtype=np.int8)

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Initialize the board
        self.board = np.zeros(11, dtype=np.int8)

        # Starting positions
        self.board[0] = 1  # Player 1 starts at position 0
        self.board[10] = 2  # Player 2 starts at position 10

        self.player_positions = {1: 0, 2: 10}

        self.current_player = 1  # Player 1 starts
        self.done = False

        return self.board.copy(), {}  # Return observation and info

    def get_opponent(self):
        return 2 if self.current_player == 1 else 1

    def step(self, action):

        if self.done:
            return self.board.copy(), 0, True, False, {}

        # Map action to (move_distance, blockade_position)
        move_distance, blockade_position = self.action_to_move(action)

        reward = 0

        # Get current position
        current_position = self.player_positions[self.current_player]

        # Determine target position based on move_distance
        direction = 1 if self.current_player == 1 else -1
        target_position = current_position + direction * move_distance

        # Check if move is valid
        if target_position < 0 or target_position > 10:
            self.done = True
            reward = -10
            return self.board.copy(), reward, self.done, False, {}

        positions_to_check = [
            current_position + direction * i for i in range(1, move_distance + 1)
        ]
        move_valid = all(self.board[pos] == 0 for pos in positions_to_check)

        if not move_valid:
            self.done = True
            reward = -10
            return self.board.copy(), reward, self.done, False, {}

        # Execute move
        self.board[current_position] = 0
        self.board[target_position] = self.current_player
        self.player_positions[self.current_player] = target_position

        # Check for victory by reaching the opponent's starting space
        if (self.current_player == 1 and target_position == 10) or (
            self.current_player == 2 and target_position == 0
        ):
            self.done = True
            reward = 1
            return self.board.copy(), reward, self.done, False, {}

        # Blockade placement
        if (
            blockade_position == 0
            or blockade_position == 10
            or self.board[blockade_position] != 0
        ):
            self.done = True
            reward = -10
            return self.board.copy(), reward, self.done, False, {}

        self.board[blockade_position] = 3

        # Check if opponent has any legal moves
        opponent = self.get_opponent()
        if not self.has_legal_moves(opponent):
            self.done = True
            reward = 1
            return self.board.copy(), reward, self.done, False, {}

        # Switch current player
        self.current_player = opponent

        return (
            self.board.copy(),
            reward,
            self.done,
            False,
            {},
        )  # Observation, reward, done, truncated, info

    def action_to_move(self, action):
        # Actions 0-8: Move 1 space, blockade positions 1-9
        # Actions 9-17: Move 2 spaces, blockade positions 1-9
        move_distance = 1 if action < 9 else 2
        blockade_position = (action % 9) + 1  # Positions 1 to 9
        return move_distance, blockade_position

    def has_legal_moves(self, player):
        current_position = self.player_positions[player]
        direction = 1 if player == 1 else -1

        for move_distance in [1, 2]:
            target_position = current_position + direction * move_distance
            if 0 <= target_position <= 10:
                positions_to_check = [
                    current_position + direction * i
                    for i in range(1, move_distance + 1)
                ]
                if all(self.board[pos] == 0 for pos in positions_to_check):
                    return True
        return False

    def valid_moves(self):
        valid_actions = []
        current_position = self.player_positions[self.current_player]
        direction = 1 if self.current_player == 1 else -1

        for action in range(18):
            move_distance, blockade_position = self.action_to_move(action)
            target_position = current_position + direction * move_distance

            if not (0 <= target_position <= 10):
                continue

            positions_to_check = [
                current_position + direction * i for i in range(1, move_distance + 1)
            ]
            if not all(self.board[pos] == 0 for pos in positions_to_check):
                continue

            if (
                blockade_position == 0
                or blockade_position == 10
                or self.board[blockade_position] != 0
            ):
                continue

            valid_actions.append(action)
        return valid_actions

    def render(self):

        positions_str = "Positions: "
        for i in range(11):
            positions_str += f"{i:<3}"

        board_str = "           "
        for i in range(11):
            if self.board[i] == 0:
                board_str += ".  "
            elif self.board[i] == 1:
                board_str += "P1 "
            elif self.board[i] == 2:
                board_str += "P2 "
            elif self.board[i] == 3:
                board_str += "X  "

        return positions_str + "\n" + board_str
