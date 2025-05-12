import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to moving to nodes 1 through 20 (indices 0-19)
        self.action_space = spaces.Discrete(20)

        # Observation space: [current_player_current_node, current_player_last_node, opponent_current_node, opponent_last_node]
        # Using low=0 to represent 'no last node' (e.g., at the start)
        self.observation_space = spaces.Box(low=0, high=20, shape=(4,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.current_player = 1  # Player 1 starts

        # Initialize player positions and last nodes
        self.player_positions = {
            1: {
                "current_node": 1,
                "last_node": 0,
            },  # last_node = 0 represents no previous node
            2: {"current_node": 1, "last_node": 0},
        }

        # Return initial observation and empty info
        return self._get_obs(), {}

    def _get_obs(self):
        cp = self.current_player
        opp = 3 - self.current_player
        obs = np.array(
            [
                self.player_positions[cp]["current_node"],
                self.player_positions[cp]["last_node"],
                self.player_positions[opp]["current_node"],
                self.player_positions[opp]["last_node"],
            ],
            dtype=np.int32,
        )
        return obs

    def step(self, action):
        if self.done:
            # Game is over
            return (
                self._get_obs(),
                0,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Map action index to node (actions 0-19 correspond to nodes 1-20)
        node = action + 1

        cp = self.current_player
        opp = 3 - self.current_player

        current_node = self.player_positions[cp]["current_node"]
        last_node = self.player_positions[cp]["last_node"]

        # Get valid moves for current player
        valid_moves = self._get_valid_moves(current_node, last_node)

        if node not in valid_moves:
            # Invalid move chosen
            self.done = True
            return (
                self._get_obs(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Valid move: update positions
        self.player_positions[cp]["last_node"] = current_node
        self.player_positions[cp]["current_node"] = node

        # Check for win condition
        if node == 20:
            self.done = True
            return self._get_obs(), 1, True, False, {}  # Current player wins

        # Switch to the next player
        self.current_player = opp
        return (
            self._get_obs(),
            0,
            False,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def _get_valid_moves(self, current_node, last_node):
        valid_moves = []

        # Factors of current_node (excluding the node itself)
        for i in range(1, current_node):
            if current_node % i == 0:
                valid_moves.append(i)

        # Multiples of current_node up to 20 (excluding the node itself)
        for i in range(current_node + 1, 21):
            if i % current_node == 0:
                valid_moves.append(i)

        # Remove last_node to prevent backtracking (if last_node is not zero)
        if last_node != 0 and last_node in valid_moves:
            valid_moves.remove(last_node)

        return valid_moves

    def valid_moves(self):
        cp = self.current_player
        current_node = self.player_positions[cp]["current_node"]
        last_node = self.player_positions[cp]["last_node"]

        valid_moves = self._get_valid_moves(current_node, last_node)
        # Convert nodes to action indices (0-19)
        valid_actions = [node - 1 for node in valid_moves]
        return valid_actions

    def render(self):
        cp = self.current_player
        opp = 3 - self.current_player
        state_str = f"Player {cp}'s Turn:\n"
        state_str += f"Current Node: {self.player_positions[cp]['current_node']}\n"
        state_str += f"Last Node: {self.player_positions[cp]['last_node']}\n"
        state_str += (
            f"Opponent's Current Node: {self.player_positions[opp]['current_node']}\n"
        )
        state_str += (
            f"Opponent's Last Node: {self.player_positions[opp]['last_node']}\n"
        )
        valid_moves = self._get_valid_moves(
            self.player_positions[cp]["current_node"],
            self.player_positions[cp]["last_node"],
        )
        state_str += f"Valid Moves: {valid_moves}\n"
        return state_str
