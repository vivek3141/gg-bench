import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: moves 1, 2, 3
        self.action_space = spaces.Discrete(
            3
        )  # Actions: 0, 1, 2 correspond to moves 1, 2, 3

        # Observation space: positions of both players
        self.observation_space = spaces.Box(low=0, high=20, shape=(2,), dtype=np.int32)

        # Initialize state
        self.positions = None
        self.current_player = None
        self.done = False

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = np.array([0, 0], dtype=np.int32)
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        return self.positions.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.positions.copy(), 0, True, False, {}  # Game is over

        # Get valid actions
        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return self.positions.copy(), reward, True, False, {}

        # Map action to move
        move = action + 1  # action 0 -> move 1, action 1 -> move 2, action 2 -> move 3

        # Get current position
        current_position = self.positions[self.current_player]

        # Perform move
        new_position = current_position + move

        # Check for trap
        if new_position in [5, 10, 15]:
            new_position = 0  # Sent back to start

        # Update position
        self.positions[self.current_player] = new_position

        # Check for win
        if new_position == 20:
            self.done = True
            reward = 1
            return self.positions.copy(), reward, True, False, {}

        # Otherwise, no reward
        reward = 0

        # Switch to next player
        self.current_player = 1 - self.current_player

        return self.positions.copy(), reward, False, False, {}

    def render(self):
        render_str = f"Player 1 is at position {self.positions[0]}.\n"
        render_str += f"Player 2 is at position {self.positions[1]}.\n"
        render_str += f"It's Player {self.current_player + 1}'s turn."
        return render_str

    def valid_moves(self):
        current_position = self.positions[self.current_player]
        valid_actions = []
        for action in range(self.action_space.n):  # actions 0, 1, 2
            move = action + 1
            if current_position + move <= 20:
                valid_actions.append(action)
        return valid_actions
