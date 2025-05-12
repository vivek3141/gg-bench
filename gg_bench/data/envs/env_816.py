import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space consists of 7 discrete actions (positions 0 to 6)
        self.action_space = spaces.Discrete(7)
        # The observation space is a Box space with 7 positions, each can be 0, 1, or 2
        self.observation_space = spaces.Box(low=0, high=2, shape=(7,), dtype=np.int32)

        # Initialize the board and game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(7, dtype=np.int32)
        self.current_player = 1  # Player A starts (1 for Player A, -1 for Player B)
        self.target_numbers = {1: 1, -1: 2}  # Player A targets 1, Player B targets 2
        self.done = False
        info = {}
        return self.board.copy(), info  # Return observation and info

    def step(self, action):
        if self.done:
            # Can't take any more steps if the game is over
            return self.board.copy(), 0, True, False, {}

        # Check if action is within the action space
        if not self.action_space.contains(action):
            # Invalid move
            self.done = True  # End the game
            reward = -10
            terminated = True
            truncated = False
            return self.board.copy(), reward, terminated, truncated, {}

        # Increment the position
        self.board[action] += 1
        if self.board[action] > 2:
            self.board[action] = 0

        # Check if current player wins
        target_number = self.target_numbers[self.current_player]
        if np.all(self.board == target_number):
            self.done = True
            reward = 1  # Winning reward
            terminated = True
            truncated = False
            return self.board.copy(), reward, terminated, truncated, {}

        # Since the game is not over, assign reward of -10 for non-winning moves as per the prompt
        reward = -10
        terminated = False
        truncated = False

        # Switch player
        self.current_player *= -1  # Switch between 1 (Player A) and -1 (Player B)

        return self.board.copy(), reward, terminated, truncated, {}

    def render(self):
        board_str = "Position: [0][1][2][3][4][5][6]\n"
        board_str += "Number:   "
        for num in self.board:
            board_str += f"[{num}]"
        board_str += "\n"
        return board_str

    def valid_moves(self):
        # All positions are valid moves since any position can be incremented
        return [i for i in range(7)]
