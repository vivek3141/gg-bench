import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, D=13):
        super(CustomEnv, self).__init__()

        # Define action space: 0 (Add 1), 1 (Multiply by 2)
        self.action_space = spaces.Discrete(2)

        # Define observation space: [N, D]
        # N: positive integer
        # D: positive integer
        self.N_LOW = 1
        self.N_HIGH = np.iinfo(np.int32).max
        self.D_LOW = 1
        self.D_HIGH = self.N_HIGH
        self.observation_space = spaces.Box(
            low=np.array([self.N_LOW, self.D_LOW]),
            high=np.array([self.N_HIGH, self.D_HIGH]),
            dtype=np.int32,
        )

        # Target Divisor D
        self.D = D  # Default value, can be passed when creating the environment

        # Initialize the environment
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Starting Number, N
        self.N = 1
        self.done = False
        self.current_player = 1

        # Allow setting D via options
        if options is not None and "D" in options:
            self.D = options["D"]

        # Return observation
        observation = np.array([self.N, self.D], dtype=np.int32)
        return observation, {}

    def step(self, action):
        # Check if done
        if self.done:
            # If the game is already over, return current observation with zero reward
            observation = np.array([self.N, self.D], dtype=np.int32)
            return observation, 0, self.done, False, {}

        # Validate action
        if action not in [0, 1]:
            # Invalid action, end the game with a penalty
            self.done = True
            observation = np.array([self.N, self.D], dtype=np.int32)
            return observation, -10, self.done, False, {"Invalid Action": action}

        # Perform action
        if action == 0:
            # Add 1
            new_N = self.N + 1
        elif action == 1:
            # Multiply by 2
            new_N = self.N * 2

        # Check if new_N overflows
        if new_N < self.N_LOW or new_N > self.N_HIGH:
            # Invalid move due to overflow, end the game with penalty
            self.done = True
            observation = np.array([self.N, self.D], dtype=np.int32)
            return observation, -10, self.done, False, {"Overflow": new_N}

        # Update N
        self.N = new_N

        # Check for win condition
        if self.N % self.D == 0:
            # Current player wins
            self.done = True
            reward = 1
            info = {"Winner": f"Player {self.current_player}"}
        else:
            # No win, negative reward per move
            reward = -10
            info = {}

        # Prepare observation
        observation = np.array([self.N, self.D], dtype=np.int32)

        # Switch player if game not over
        if not self.done:
            self.current_player = 2 if self.current_player == 1 else 1

        # Return observation, reward, done, truncated=False, info
        return observation, reward, self.done, False, info

    def render(self):
        # Return a string representing the current state
        state_str = f"Current N: {self.N}\n"
        state_str += f"Current Player: Player {self.current_player}\n"
        state_str += f"Target Divisor D: {self.D}\n"
        return state_str

    def valid_moves(self):
        # Both actions are valid unless they cause overflow
        valid_actions = []
        # Check action 0: Add 1
        if self.N + 1 <= self.N_HIGH:
            valid_actions.append(0)
        # Check action 1: Multiply by 2
        if self.N * 2 <= self.N_HIGH:
            valid_actions.append(1)
        return valid_actions
