import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 19 possible actions:
        # Action 0: Move Forward
        # Actions 1-9: Place Block at positions 1-9
        # Actions 10-18: Remove own Block from positions 1-9
        self.action_space = spaces.Discrete(19)

        # Observation space represents the board positions 0 to 10
        # Possible values:
        # 0: Empty
        # 1: Player 1's Token
        # -1: Player 2's Token
        # 2: Block placed by Player 1
        # -2: Block placed by Player 2
        self.observation_space = spaces.Box(low=-2, high=2, shape=(11,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(11, dtype=np.int8)  # positions 0 to 10
        self.board[0] = 1  # Player 1's token at position 0
        self.board[10] = -1  # Player 2's token at position 10

        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.tokens = {1: 0, -1: 10}  # positions of Player 1 and Player 2 tokens

        self.blocks = {
            1: set(),
            -1: set(),
        }  # sets of positions of blocks placed by each player

        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        reward = 0

        player = self.current_player
        opponent = -self.current_player
        token_pos = self.tokens[player]

        invalid_move = False

        if action == 0:
            # Move Forward
            direction = 1 if player == 1 else -1
            next_pos = token_pos + direction
            if 0 <= next_pos <= 10:
                if self.board[next_pos] == 0:
                    # Move to next position
                    self.board[token_pos] = 0
                    self.board[next_pos] = player
                    self.tokens[player] = next_pos
                    # Check for win
                    if (player == 1 and next_pos == 10) or (
                        player == -1 and next_pos == 0
                    ):
                        self.done = True
                        reward = 1  # Win
                else:
                    # Cannot move onto a blocked position or occupied position
                    invalid_move = True
            else:
                # Cannot move outside the board
                invalid_move = True
        elif 1 <= action <= 9:
            # Place Block at position action (positions 1 to 9)
            block_pos = action
            if (
                self.board[block_pos] == 0
                and block_pos != self.tokens[1]
                and block_pos != self.tokens[-1]
            ):
                # Place block
                self.board[block_pos] = (
                    2 * player
                )  # 2 for Player 1's block, -2 for Player 2's block
                self.blocks[player].add(block_pos)
            else:
                invalid_move = True
        elif 10 <= action <= 18:
            # Remove own Block at position action - 9 (positions 1 to 9)
            block_pos = action - 9
            if block_pos in self.blocks[player]:
                # Remove block
                self.board[block_pos] = 0
                self.blocks[player].remove(block_pos)
            else:
                invalid_move = True
        else:
            invalid_move = True

        if invalid_move:
            reward = -10
            self.done = True
            return self.board.copy(), reward, True, False, {}

        # Switch player if the game is not over
        if not self.done:
            self.current_player *= -1

        return self.board.copy(), reward, self.done, False, {}

    def render(self):
        board_str = ""
        for i in range(11):
            pos = self.board[i]
            if pos == 0:
                content = "  "
            elif pos == 1:
                content = "P1"
            elif pos == -1:
                content = "P2"
            elif pos == 2 or pos == -2:
                content = "XX"
            else:
                content = "??"
            board_str += f"[{content}]"
        return board_str

    def valid_moves(self):
        valid_actions = []
        player = self.current_player
        token_pos = self.tokens[player]

        # Check if Move Forward is possible
        direction = 1 if player == 1 else -1
        next_pos = token_pos + direction
        if 0 <= next_pos <= 10 and self.board[next_pos] == 0:
            valid_actions.append(0)

        # Check for possible block placements
        for pos in range(1, 10):
            if (
                self.board[pos] == 0
                and pos != self.tokens[1]
                and pos != self.tokens[-1]
            ):
                valid_actions.append(pos)  # positions 1-9 correspond to actions 1-9

        # Check for possible block removals (own blocks)
        for pos in self.blocks[player]:
            action = pos + 9  # positions 1-9 correspond to actions 10-18
            valid_actions.append(action)

        return valid_actions
