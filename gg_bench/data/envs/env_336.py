import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 0 (pass), 1-75 (allocations)
        self.action_space = spaces.Discrete(76)

        # Observation space: Allocations and remaining energy for both players
        # Allocations: 5 nodes x 2 players = 10
        # Remaining energy: 2 players = 2
        self.observation_space = spaces.Box(low=0, high=15, shape=(12,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.allocations = np.zeros((2, 5), dtype=np.int32)  # allocations[player][node]
        self.remaining_energy = np.array([15, 15], dtype=np.int32)
        self.current_player = 0  # Player 0 and Player 1
        self.passed = [False, False]
        self.done = False
        self.in_sudden_death = False
        self.neutral_nodes = []  # Nodes that are neutral after a tie
        return self._get_obs(), {}

    def _get_obs(self):
        # Flatten allocations and include remaining energy
        obs = np.concatenate((self.allocations.flatten(), self.remaining_energy))
        return obs

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        if action >= self.action_space.n or action < 0:
            return self._get_obs(), -10, True, False, {}

        if self.passed[self.current_player]:
            # Player has already passed, cannot make more allocations
            return self._get_obs(), -10, True, False, {}

        if action == 0:
            # Pass
            self.passed[self.current_player] = True
        else:
            # Decode action
            node_index = (action - 1) // 15
            units_allocated = ((action - 1) % 15) + 1

            # If in sudden death, check if node is neutral
            if self.in_sudden_death and node_index not in self.neutral_nodes:
                return self._get_obs(), -10, True, False, {}

            if units_allocated > self.remaining_energy[self.current_player]:
                # Invalid move, not enough energy
                return self._get_obs(), -10, True, False, {}

            self.allocations[self.current_player, node_index] += units_allocated
            self.remaining_energy[self.current_player] -= units_allocated

        # Check if both players have passed
        if all(self.passed):
            # Reveal phase
            rewards = self._reveal_phase()
            if self.done:
                return self._get_obs(), rewards[self.current_player], True, False, {}
            else:
                # Reset for next allocation phase (Sudden Death Round)
                self.passed = [False, False]
                self.current_player = 0
                return self._get_obs(), 0, False, False, {}
        else:
            # Switch player
            self.current_player = 1 - self.current_player
            return self._get_obs(), 0, False, False, {}

    def _reveal_phase(self):
        # Determine winner
        allocations_p0 = self.allocations[0]
        allocations_p1 = self.allocations[1]

        nodes_won_p0 = (allocations_p0 > allocations_p1).sum()
        nodes_won_p1 = (allocations_p1 > allocations_p0).sum()
        neutral_nodes_mask = allocations_p0 == allocations_p1
        self.neutral_nodes = np.where(neutral_nodes_mask)[0]

        if nodes_won_p0 >= 3:
            # Player 0 wins
            self.done = True
            return [1, -1]
        elif nodes_won_p1 >= 3:
            # Player 1 wins
            self.done = True
            return [-1, 1]
        else:
            if nodes_won_p0 > nodes_won_p1:
                self.done = True
                return [1, -1]
            elif nodes_won_p1 > nodes_won_p0:
                self.done = True
                return [-1, 1]
            else:
                # Tie, proceed to Sudden Death Round
                self._initiate_sudden_death()
                return [0, 0]  # No reward yet

    def _initiate_sudden_death(self):
        self.in_sudden_death = True
        # Reset allocations for neutral nodes
        self.allocations[:, self.neutral_nodes] = 0
        # Each player receives additional 5 energy units
        self.remaining_energy = np.array([5, 5], dtype=np.int32)
        # Only neutral nodes are contested
        # Action space remains the same; step() will check for valid nodes
        # Passed flags reset
        self.passed = [False, False]

    def render(self):
        lines = []
        for i in range(5):
            line = f"Node {i+1}: Player 0 allocated {self.allocations[0][i]} units, Player 1 allocated {self.allocations[1][i]} units"
            lines.append(line)
        info = "\n".join(lines)
        return info

    def valid_moves(self):
        if self.done:
            return []

        if self.passed[self.current_player]:
            return [0]  # Only option is to pass
        else:
            valid_actions = [0]  # Always can pass
            remaining_energy = self.remaining_energy[self.current_player]
            max_units = min(15, remaining_energy)
            if max_units <= 0:
                return [0]
            if self.in_sudden_death:
                valid_nodes = self.neutral_nodes
            else:
                valid_nodes = range(5)
            for node_index in valid_nodes:
                for units in range(1, max_units + 1):
                    action = 1 + node_index * 15 + (units - 1)
                    valid_actions.append(action)
            return valid_actions
