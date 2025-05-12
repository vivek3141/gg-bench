import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define node labels and mappings
        self.node_labels = [1, 2, 3, 4, 6, 8, 12]
        self.num_nodes = len(self.node_labels)
        self.node_indices = list(range(self.num_nodes))
        self.label_to_index = {
            label: index for index, label in enumerate(self.node_labels)
        }
        self.index_to_label = {
            index: label for index, label in enumerate(self.node_labels)
        }

        # Define adjacency list (connections between nodes)
        self.adj_list = {
            0: [1],  # Node 1 connected to Node 2
            1: [0, 2, 3],  # Node 2 connected to Nodes 1, 3, 4
            2: [1, 4],  # Node 3 connected to Nodes 2, 6
            3: [1, 5],  # Node 4 connected to Nodes 2, 8
            4: [2, 6],  # Node 6 connected to Nodes 3, 12
            5: [3, 6],  # Node 8 connected to Nodes 4, 12
            6: [4, 5],  # Node 12 connected to Nodes 6, 8
        }

        # Define action and observation space
        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.Box(low=1, high=12, shape=(2,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1  # Player 1 starts
        self.player_positions = {
            1: 0,
            2: 0,
        }  # Both players start at node index 0 (label 1)
        self.done = False

        # Observation: [current_player_node_label, opponent_node_label]
        observation = np.array(
            [
                self.node_labels[self.player_positions[self.current_player]],
                self.node_labels[
                    self.player_positions[2 if self.current_player == 1 else 1]
                ],
            ],
            dtype=np.int32,
        )

        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                np.array(
                    [
                        self.node_labels[self.player_positions[self.current_player]],
                        self.node_labels[
                            self.player_positions[2 if self.current_player == 1 else 1]
                        ],
                    ],
                    dtype=np.int32,
                ),
                0,
                True,
                False,
                {},
            )

        # Get current node index and action node index
        current_node_index = self.player_positions[self.current_player]
        action_node_index = action

        # Check if action node is valid
        if action_node_index not in self.node_indices:
            # Invalid node index
            reward = -10
            self.done = True
            return (
                np.array(
                    [
                        self.node_labels[current_node_index],
                        self.node_labels[
                            self.player_positions[2 if self.current_player == 1 else 1]
                        ],
                    ],
                    dtype=np.int32,
                ),
                reward,
                True,
                False,
                {},
            )

        # Check if action node is connected to current node
        if action_node_index not in self.adj_list[current_node_index]:
            # Invalid move (not connected)
            reward = -10
            self.done = True
            return (
                np.array(
                    [
                        self.node_labels[current_node_index],
                        self.node_labels[
                            self.player_positions[2 if self.current_player == 1 else 1]
                        ],
                    ],
                    dtype=np.int32,
                ),
                reward,
                True,
                False,
                {},
            )

        # Check movement rules
        current_label = self.node_labels[current_node_index]
        dest_label = self.node_labels[action_node_index]

        if dest_label % current_label != 0 and current_label % dest_label != 0:
            # Invalid move (movement rules not satisfied)
            reward = -10
            self.done = True
            return (
                np.array(
                    [
                        current_label,
                        self.node_labels[
                            self.player_positions[2 if self.current_player == 1 else 1]
                        ],
                    ],
                    dtype=np.int32,
                ),
                reward,
                True,
                False,
                {},
            )

        # Valid move
        self.player_positions[self.current_player] = action_node_index

        # Check for win condition
        if action_node_index == self.label_to_index[12]:  # Reached End Node (label 12)
            reward = 1
            self.done = True
            return (
                np.array(
                    [
                        dest_label,
                        self.node_labels[
                            self.player_positions[2 if self.current_player == 1 else 1]
                        ],
                    ],
                    dtype=np.int32,
                ),
                reward,
                True,
                False,
                {},
            )

        # Continue game
        reward = 0
        # Swap current player
        self.current_player = 2 if self.current_player == 1 else 1

        observation = np.array(
            [
                self.node_labels[self.player_positions[self.current_player]],
                self.node_labels[
                    self.player_positions[2 if self.current_player == 1 else 1]
                ],
            ],
            dtype=np.int32,
        )

        return observation, reward, False, False, {}

    def render(self):
        board_str = "Players' Positions:\n"
        for index in self.node_indices:
            node_label = self.node_labels[index]
            occupants = []
            for player, position in self.player_positions.items():
                if position == index:
                    occupants.append(f"Player {player}")
            occupants_str = ", ".join(occupants) if occupants else "None"
            board_str += f"Node {node_label}: {occupants_str}\n"
        return board_str

    def valid_moves(self):
        # Get current node index
        current_node_index = self.player_positions[self.current_player]
        current_label = self.node_labels[current_node_index]

        valid_moves = []
        for neighbor in self.adj_list[current_node_index]:
            dest_label = self.node_labels[neighbor]
            # Check movement rules
            if dest_label % current_label == 0 or current_label % dest_label == 0:
                valid_moves.append(neighbor)
        return valid_moves
