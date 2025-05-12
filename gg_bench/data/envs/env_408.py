import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0, 1, 2 corresponding to moving 1, 2, or 3 steps respectively
        self.action_space = spaces.Discrete(3)

        # Observation space: [p1_position, p2_position, p1_skip_turn, p2_skip_turn, current_player]
        self.observation_space = spaces.Box(low=0, high=21, shape=(5,), dtype=np.int32)

        # Initialize the game state
        self.positions = [0, 0]  # Positions of Player 1 and Player 2
        self.skip_turns = [0, 0]  # Skip-turn counters for both players
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = [0, 0]
        self.skip_turns = [0, 0]
        self.current_player = 0  # Player 1 starts
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if current player needs to skip turn
        if self.skip_turns[self.current_player] > 0:
            self.skip_turns[self.current_player] -= 1
            # Skip overrides pass; proceed to the next player
            self.current_player = 1 - self.current_player
            observation = self._get_observation()
            return observation, 0, False, False, {}

        # Get valid moves
        valid_actions = self.valid_moves()
        if len(valid_actions) == 0:
            # No valid moves; player must pass turn
            self.current_player = 1 - self.current_player
            observation = self._get_observation()
            return observation, 0, False, False, {}

        # Check if action is valid
        if action not in valid_actions:
            # Invalid move
            self.done = True
            observation = self._get_observation()
            return observation, -10, True, False, {}

        # Apply move
        move = action + 1
        new_position = self.positions[self.current_player] + move

        # Update position
        self.positions[self.current_player] = new_position

        # Check for skip-turn positions
        if new_position in [5, 10, 15, 20]:
            self.skip_turns[self.current_player] += 1

        # Check for victory
        if new_position == 20:
            self.done = True
            observation = self._get_observation()
            return observation, 1, True, False, {}

        # Switch to the other player
        self.current_player = 1 - self.current_player
        observation = self._get_observation()
        return observation, 0, False, False, {}

    def render(self):
        board = ""
        for pos in range(1, 21):
            token = ""
            if self.positions[0] == pos and self.positions[1] == pos:
                token = "X/O"
            elif self.positions[0] == pos:
                token = "X"
            elif self.positions[1] == pos:
                token = "O"
            else:
                token = str(pos)
            board += f"{token:>5}"
        board += f"\n\nPlayer 1 (X) Position: {self.positions[0]}, Skip Turns: {self.skip_turns[0]}"
        board += f"\nPlayer 2 (O) Position: {self.positions[1]}, Skip Turns: {self.skip_turns[1]}"
        board += f"\nCurrent Turn: {'Player 1 (X)' if self.current_player == 0 else 'Player 2 (O)'}"
        return board

    def valid_moves(self):
        position = self.positions[self.current_player]
        valid_actions = []
        for action in range(3):  # Actions 0, 1, 2
            move = action + 1
            new_position = position + move
            if new_position <= 20:
                valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        return np.array(
            [
                self.positions[0],
                self.positions[1],
                self.skip_turns[0],
                self.skip_turns[1],
                self.current_player,
            ],
            dtype=np.int32,
        )
