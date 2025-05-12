import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Each possible action is moving a controlled stack from a position (0-10)
        # by 1 or 2 squares forward. There are 11 positions and 2 possible moves (1 or 2 squares)
        # per position, so total actions = 11 positions * 2 move options = 22 actions.
        self.action_space = spaces.Discrete(22)

        # Observation space: The board has 11 positions, each can have a stack of tokens.
        # Each stack can have a maximum of 6 tokens (total initial tokens of both players).
        # We'll represent the board as a (11, 6) array where each cell contains 0 (empty),
        # 1 (Player 1's token), or 2 (Player 2's token).
        self.observation_space = spaces.Box(low=0, high=2, shape=(11, 6), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board as a list of lists representing stacks at each position
        self.board = [[] for _ in range(11)]  # Positions 0 to 10

        # Place Player 1's tokens on positions 0, 1, and 2
        self.board[0] = [1]
        self.board[1] = [1]
        self.board[2] = [1]

        # Place Player 2's tokens on positions 8, 9, and 10
        self.board[8] = [2]
        self.board[9] = [2]
        self.board[10] = [2]

        self.current_player = 1  # Player 1 starts
        self.done = False

        return self.get_observation(), {}  # Return observation and info

    def get_observation(self):
        obs = np.zeros((11, 6), dtype=np.int8)
        for pos in range(11):
            stack = self.board[pos]
            for level, token in enumerate(stack):
                obs[pos, level] = token
        return obs

    def step(self, action):
        if self.done:
            return self.get_observation(), -10, True, False, {}

        # Decode action
        position = action // 2  # Position to move from (0-10)
        move_distance = (action % 2) + 1  # Move forward 1 or 2 squares

        # Validate action
        if not self.board[position]:
            # No stack at the position
            self.done = True
            return self.get_observation(), -10, True, False, {}
        else:
            top_token = self.board[position][-1]
            if top_token != self.current_player:
                # Player does not control this stack
                self.done = True
                return self.get_observation(), -10, True, False, {}

        # Calculate new position
        if self.current_player == 1:
            new_position = position + move_distance
        else:  # self.current_player == 2
            new_position = position - move_distance

        # Check if new position is within bounds
        if new_position < 0 or new_position > 10:
            self.done = True
            return self.get_observation(), -10, True, False, {}

        # Execute move
        moving_stack = self.board[position]
        self.board[position] = []  # Remove stack from original position

        if self.board[new_position]:
            # Existing stack at new position, place moving stack on top
            self.board[new_position] = self.board[new_position] + moving_stack
        else:
            # Empty position
            self.board[new_position] = moving_stack

        # Handle capturing opponent's tokens if moved into own territory
        home_territory_p1 = [0, 1, 2, 3, 4]  # Player 1's home territory
        home_territory_p2 = [6, 7, 8, 9, 10]  # Player 2's home territory

        if self.current_player == 1 and new_position in home_territory_p1:
            self.capture_opponent_tokens(new_position)
        elif self.current_player == 2 and new_position in home_territory_p2:
            self.capture_opponent_tokens(new_position)

        # Check for winning condition
        if self.current_player == 1 and new_position == 10:
            self.done = True
            return self.get_observation(), 1, True, False, {}
        elif self.current_player == 2 and new_position == 0:
            self.done = True
            return self.get_observation(), 1, True, False, {}

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1
        return self.get_observation(), -10, False, False, {}

    def capture_opponent_tokens(self, position):
        # Remove opponent's tokens from the stack at the given position
        stack = self.board[position]
        new_stack = [token for token in stack if token == self.current_player]
        self.board[position] = new_stack

    def render(self):
        board_str = "[Player 1 Base] "
        for pos in range(11):
            stack = self.board[pos]
            if stack:
                stack_str = "".join(["P1" if token == 1 else "P2" for token in stack])
                board_str += f"{pos}:[{stack_str}] "
            else:
                board_str += f"{pos} "
        board_str += "[Player 2 Base]"
        return board_str

    def valid_moves(self):
        moves = []
        for position in range(11):
            if self.board[position]:
                top_token = self.board[position][-1]
                if top_token == self.current_player:
                    for move_distance in [1, 2]:
                        if self.current_player == 1:
                            new_position = position + move_distance
                        else:
                            new_position = position - move_distance
                        if 0 <= new_position <= 10:
                            moves.append(position * 2 + (move_distance - 1))
        return moves
