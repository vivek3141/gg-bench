import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is Discrete(20) for numbers 1 to 20
        self.action_space = spaces.Discrete(20)
        # The observation is a 20-length vector with values:
        # 0: Available
        # -1: Unavailable (adjacent to selected numbers)
        # 1: Selected by Player 1
        # 2: Selected by Player 2
        self.observation_space = spaces.Box(low=-1, high=2, shape=(20,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(20, dtype=np.int8)  # All numbers are initially available
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.state.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.state.copy(), 0, True, False, {}  # Game is already over

        valid_moves = self.valid_moves()
        if action not in valid_moves:
            self.done = True
            return (
                self.state.copy(),
                -10,
                True,
                False,
                {},
            )  # Invalid move, game over

        # Mark the chosen number as selected by the current player
        self.state[action] = self.current_player  # 1 for Player 1, 2 for Player 2

        # Update adjacent numbers to be unavailable
        if action > 0 and self.state[action - 1] == 0:
            self.state[action - 1] = -1  # Unavailable
        if action < 19 and self.state[action + 1] == 0:
            self.state[action + 1] = -1  # Unavailable

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if the next player has any valid moves
        if not self.valid_moves():
            # The previous player wins
            self.done = True
            # Switch back to the winning player for correct reward attribution
            winner = 2 if self.current_player == 1 else 1
            return self.state.copy(), 1, True, False, {}  # Current player wins

        return self.state.copy(), 0, False, False, {}  # Continue game

    def render(self):
        # Create a string representation of the game state
        state_str = f"Current Player: {self.current_player}\n"
        state_str += "Available Numbers: "
        available = [str(i + 1) for i, x in enumerate(self.state) if x == 0]
        state_str += ", ".join(available) + "\n"

        state_str += "Selected by Player 1: "
        selected_p1 = [str(i + 1) for i, x in enumerate(self.state) if x == 1]
        state_str += ", ".join(selected_p1) + "\n"

        state_str += "Selected by Player 2: "
        selected_p2 = [str(i + 1) for i, x in enumerate(self.state) if x == 2]
        state_str += ", ".join(selected_p2) + "\n"

        state_str += "Unavailable Numbers: "
        unavailable = [str(i + 1) for i, x in enumerate(self.state) if x == -1]
        state_str += ", ".join(unavailable) + "\n"

        return state_str

    def valid_moves(self):
        # Return a list of indices of valid moves (numbers that are available)
        return [i for i, x in enumerate(self.state) if x == 0]
