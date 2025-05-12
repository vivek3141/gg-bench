import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0,1,2 representing moving forward by 1,2,3 steps
        self.action_space = spaces.Discrete(3)

        # Observation: an array of length 11, values 0 (empty), 1 (Player 1), 2 (Player 2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(11,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board
        self.board = np.zeros(11, dtype=np.int8)
        self.board[0] = 1  # Player 1 starts at position 0
        self.board[10] = 2  # Player 2 starts at position 10

        self.player_positions = {1: 0, 2: 10}
        self.current_player = 1  # Player 1 starts
        self.done = False

        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        move_distance = action + 1  # action 0->move 1, action1->move2, action2->move3
        player_pos = self.player_positions[self.current_player]

        if self.current_player == 1:
            new_pos = player_pos + move_distance
            opponent = 2
        else:
            new_pos = player_pos - move_distance
            opponent = 1

        opponent_pos = self.player_positions[opponent]

        # Check move validity
        if (self.current_player == 1 and (new_pos > 10 or new_pos == opponent_pos)) or (
            self.current_player == 2 and (new_pos < 0 or new_pos == opponent_pos)
        ):
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Move is valid
        self.board[player_pos] = 0  # Remove player from current position
        self.board[new_pos] = self.current_player  # Place player at new position
        self.player_positions[self.current_player] = new_pos

        # Check for pushing
        if self.current_player == 1 and new_pos == opponent_pos - 1:
            # Push opponent back
            new_opponent_pos = max(opponent_pos - 1, 0)
            self.board[opponent_pos] = 0
            self.board[new_opponent_pos] = opponent
            self.player_positions[opponent] = new_opponent_pos
        elif self.current_player == 2 and new_pos == opponent_pos + 1:
            # Push opponent back
            new_opponent_pos = min(opponent_pos + 1, 10)
            self.board[opponent_pos] = 0
            self.board[new_opponent_pos] = opponent
            self.player_positions[opponent] = new_opponent_pos

        # Check for win
        if (self.current_player == 1 and new_pos == 10) or (
            self.current_player == 2 and new_pos == 0
        ):
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch player
        self.current_player = opponent

        return self.board.copy(), 0, False, False, {}

    def render(self):
        board_str = ""
        for i in range(11):
            if self.board[i] == 0:
                board_str += "[  _ ]"
            elif self.board[i] == 1:
                board_str += "[ P1 ]"
            elif self.board[i] == 2:
                board_str += "[ P2 ]"
        board_str += "\n"
        for i in range(11):
            board_str += f"  {i}  "
        return board_str

    def valid_moves(self):
        valid_actions = []
        for action in range(3):  # Actions 0, 1, 2 correspond to moving 1, 2, 3 steps
            move_distance = action + 1
            player_pos = self.player_positions[self.current_player]
            opponent = 2 if self.current_player == 1 else 1
            opponent_pos = self.player_positions[opponent]

            if self.current_player == 1:
                new_pos = player_pos + move_distance
                if new_pos > 10 or new_pos == opponent_pos:
                    continue
                valid_actions.append(action)
            else:
                new_pos = player_pos - move_distance
                if new_pos < 0 or new_pos == opponent_pos:
                    continue
                valid_actions.append(action)
        return valid_actions
