import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to moving to Nodes 1 to 30 (indices 0 to 29)
        self.action_space = spaces.Discrete(30)

        # Observation space consists of:
        # - Positions of the two players (values from 1 to 30)
        # - Current player index (1 or 2)
        self.observation_space = spaces.Box(
            low=1, high=30, shape=(3,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = [1, 1]  # Positions of Player 1 and Player 2
        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_moves = self.valid_moves()
        if action not in valid_moves:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        target_node = action + 1  # Nodes are numbered from 1 to 30
        opponent = 2 if self.current_player == 1 else 1

        # Update current player's position
        self.positions[self.current_player - 1] = target_node

        # Check for win condition
        if target_node == 30:
            # Current player wins by reaching Node 30
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch to next player
        self.current_player = opponent

        # Check if the next player has any valid moves
        if not self.valid_moves():
            # Next player has no valid moves; current player wins
            self.done = True
            # Switch back to the player who wins
            self.current_player = opponent
            return self._get_observation(), 1, True, False, {}

        # Continue game
        return self._get_observation(), 0, False, False, {}

    def render(self):
        # Return a visual representation of the environment state as a string
        board_str = "Game State:\n"
        for i in range(1, 31):
            occupant = None
            if self.positions[0] == i and self.positions[1] == i:
                occupant = "Player1 & Player2"
            elif self.positions[0] == i:
                occupant = "Player1"
            elif self.positions[1] == i:
                occupant = "Player2"
            else:
                occupant = "Empty"
            board_str += f"Node {i}: {occupant}\n"
        board_str += f"Current Player: Player {self.current_player}\n"
        return board_str

    def valid_moves(self):
        current_pos = self.positions[self.current_player - 1]
        opponent_pos = self.positions[2 - self.current_player - 1]
        valid_moves = []

        if current_pos == 1:
            # From Node 1, can move to any unoccupied node ahead
            for node in range(2, 31):
                if node != opponent_pos:
                    valid_moves.append(node - 1)  # Actions are indices
        else:
            # Multiples of current position
            multiples = [i for i in range(current_pos * 2, 31, current_pos)]
            # Factors of current position
            factors = [i for i in range(1, current_pos) if current_pos % i == 0]

            candidate_nodes = set(multiples + factors)
            for node in candidate_nodes:
                if node > current_pos and node != opponent_pos:
                    valid_moves.append(node - 1)  # Actions are indices

        return valid_moves

    def _get_observation(self):
        # Return the observation: [current player's position, opponent's position, current player index]
        current_player_pos = self.positions[self.current_player - 1]
        opponent_pos = self.positions[2 - self.current_player - 1]
        observation = np.array(
            [current_player_pos, opponent_pos, self.current_player], dtype=np.float32
        )
        return observation
