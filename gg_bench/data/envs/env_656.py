import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 8 possible multipliers from 2 to 9
        self.action_space = spaces.Discrete(8)

        # Observation space consists of:
        # - Current player: -1 or 1
        # - Running total: from 1 to maximum possible total (362880)
        # - Availability of multipliers 2-9: 1 (available) or 0 (used)
        low_obs = np.array(
            [-1.0, 1.0] + [0.0] * 8,
            dtype=np.float32,
        )
        high_obs = np.array(
            [1.0, 362880.0] + [1.0] * 8,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32
        )

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1  # Player 1 starts
        self.running_total = 1
        self.available_multipliers = [1] * 8  # Multipliers 2-9 available
        self.done = False

        # Construct the initial observation
        observation = np.array(
            [self.current_player, self.running_total] + self.available_multipliers,
            dtype=np.float32,
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        # Check if the game is already over
        if self.done:
            observation = np.array(
                [self.current_player, self.running_total] + self.available_multipliers,
                dtype=np.float32,
            )
            return observation, -10, True, False, {}

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            observation = np.array(
                [self.current_player, self.running_total] + self.available_multipliers,
                dtype=np.float32,
            )
            return observation, -10, True, False, {}

        # Map action to multiplier
        action_to_multiplier = [2, 3, 4, 5, 6, 7, 8, 9]
        multiplier = action_to_multiplier[action]

        # Update the running total
        self.running_total *= multiplier

        # Update the multiplier availability
        self.available_multipliers[action] = 0

        # Check for win condition
        if self.running_total >= 100:
            self.done = True
            observation = np.array(
                [self.current_player, self.running_total] + self.available_multipliers,
                dtype=np.float32,
            )
            return observation, 1, True, False, {}  # Current player wins

        # Check if all multipliers are used
        if sum(self.available_multipliers) == 0:
            self.done = True
            # Determine the winner based on the running total
            # Since both players have used all multipliers without reaching 100
            # The player with the higher running total wins
            # In this implementation, the current player wins if they have the highest total
            # Since the running total is shared, the current player loses if the total is less than 100
            observation = np.array(
                [self.current_player, self.running_total] + self.available_multipliers,
                dtype=np.float32,
            )
            return observation, -10, True, False, {}

        # Switch the current player
        self.current_player *= -1

        # Construct the observation
        observation = np.array(
            [self.current_player, self.running_total] + self.available_multipliers,
            dtype=np.float32,
        )
        return observation, -10, False, False, {}

    def render(self):
        # Return a string representation of the game state
        player_str = "Player 1" if self.current_player == 1 else "Player 2"
        multipliers = [2, 3, 4, 5, 6, 7, 8, 9]
        available_multipliers = [
            str(m)
            for m, available in zip(multipliers, self.available_multipliers)
            if available
        ]
        used_multipliers = [
            str(m)
            for m, available in zip(multipliers, self.available_multipliers)
            if not available
        ]
        state_str = f"Current Player: {player_str}\n"
        state_str += f"Running Total: {self.running_total}\n"
        state_str += f"Available Multipliers: {', '.join(available_multipliers)}\n"
        state_str += f"Used Multipliers: {', '.join(used_multipliers)}\n"
        return state_str

    def valid_moves(self):
        # Return the list of valid action indices
        return [
            i for i, available in enumerate(self.available_multipliers) if available
        ]
