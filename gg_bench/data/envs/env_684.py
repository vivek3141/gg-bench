import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Add 1, 1 - Add 2, 2 - Add 3, 3 - Multiply by 2, 4 - Multiply by 3
        self.action_space = spaces.Discrete(5)

        # Observations: Player 1 position, Player 2 position, Current player's turn (0 or 1)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), high=np.array([20, 20, 1]), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_positions = [0, 0]  # Positions of Player 1 and Player 2
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, return the current observation
            return self._get_observation(), 0, True, False, {}

        # Check if the current player has any valid moves
        valid_actions = self.valid_moves()
        if not valid_actions:
            # Current player has no valid moves and loses the game
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        if action not in valid_actions:
            # Invalid move, current player loses the game
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Apply the action
        current_position = self.player_positions[self.current_player]
        new_position = self._apply_action(current_position, action)

        if new_position > 20 or new_position <= current_position:
            # Move is invalid, current player loses the game
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Update the current player's position
        self.player_positions[self.current_player] = new_position

        if new_position == 20:
            # Current player wins the game
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, {}

        # Switch to the other player
        self.current_player = 1 - self.current_player
        reward = 0
        return self._get_observation(), reward, False, False, {}

    def render(self):
        # Return a string representation of the game state
        return (
            f"Player 1 is on Space {self.player_positions[0]}, "
            f"Player 2 is on Space {self.player_positions[1]}, "
            f"It is Player {self.current_player + 1}'s turn."
        )

    def valid_moves(self):
        # Return a list of valid actions for the current player
        position = self.player_positions[self.current_player]
        valid_actions = []
        for action in range(self.action_space.n):
            new_position = self._apply_action(position, action)
            if new_position > position and new_position <= 20:
                valid_actions.append(action)
        return valid_actions

    def _apply_action(self, position, action):
        # Apply the action to the given position and return the new position
        if action == 0:
            # Add 1
            return position + 1
        elif action == 1:
            # Add 2
            return position + 2
        elif action == 2:
            # Add 3
            return position + 3
        elif action == 3:
            # Multiply by 2
            return position * 2
        elif action == 4:
            # Multiply by 3
            return position * 3

    def _get_observation(self):
        # Return the current observation
        return np.array(
            [self.player_positions[0], self.player_positions[1], self.current_player],
            dtype=np.int32,
        )
