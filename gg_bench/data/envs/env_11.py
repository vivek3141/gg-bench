import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Actions: 0-8
        self.observation_space = spaces.MultiBinary(
            6
        )  # 6 elements (F, W, E) for both players

        # Outcome matrix:
        # Rows: Attacker's element index (0: F, 1: W, 2: E)
        # Columns: Defender's element index (0: F, 1: W, 2: E)
        # Values:
        #   1  : Attacker captures defender's element
        #  -1  : Defender captures attacker's element
        #   0  : Both elements are captured
        self.outcome_matrix = np.array(
            [
                [0, -1, 1],  # Fire (F) vs [F, W, E]
                [1, 0, -1],  # Water (W) vs [F, W, E]
                [-1, 1, 0],  # Earth (E) vs [F, W, E]
            ]
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Each player's elements: [F, W, E], where 1 indicates the element is present
        self.players_elements = [
            np.array([1, 1, 1], dtype=int),  # Player 1
            np.array([1, 1, 1], dtype=int),  # Player 2
        ]
        self.current_player = 0  # Player 1 starts
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        # Observation: Concatenated elements of both players
        return np.concatenate(self.players_elements)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        my_elements = self.players_elements[self.current_player]
        opp_player = 1 - self.current_player
        opp_elements = self.players_elements[opp_player]

        my_element_idx = action // 3  # 0 (F), 1 (W), 2 (E)
        opp_element_idx = action % 3  # 0 (F), 1 (W), 2 (E)

        # Check if both elements are available
        if my_elements[my_element_idx] == 0 or opp_elements[opp_element_idx] == 0:
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Resolve the attack
        outcome = self.outcome_matrix[my_element_idx][opp_element_idx]

        if outcome == 1:
            # Attacker captures defender's element
            opp_elements[opp_element_idx] = 0
        elif outcome == -1:
            # Defender captures attacker's element
            my_elements[my_element_idx] = 0
        elif outcome == 0:
            # Both elements are captured
            my_elements[my_element_idx] = 0
            opp_elements[opp_element_idx] = 0

        # Update players' elements
        self.players_elements[self.current_player] = my_elements
        self.players_elements[opp_player] = opp_elements

        # Check for win conditions
        if np.sum(opp_elements) == 0:
            # Current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}
        if np.sum(my_elements) == 0:
            # Opponent wins
            self.done = True
            return self._get_obs(), 0, True, False, {}

        # Switch to the next player
        self.current_player = opp_player
        return self._get_obs(), 0, False, False, {}

    def render(self):
        element_names = ["F", "W", "E"]
        p1_elements = [
            element_names[i] for i in range(3) if self.players_elements[0][i] == 1
        ]
        p2_elements = [
            element_names[i] for i in range(3) if self.players_elements[1][i] == 1
        ]
        output = f"Current player: Player {self.current_player + 1}\n"
        output += (
            f"Player 1's elements: {' '.join(p1_elements) if p1_elements else 'None'}\n"
        )
        output += (
            f"Player 2's elements: {' '.join(p2_elements) if p2_elements else 'None'}\n"
        )
        return output

    def valid_moves(self):
        moves = []
        my_elements = self.players_elements[self.current_player]
        opp_player = 1 - self.current_player
        opp_elements = self.players_elements[opp_player]
        for action in range(9):
            my_element_idx = action // 3
            opp_element_idx = action % 3
            if my_elements[my_element_idx] == 1 and opp_elements[opp_element_idx] == 1:
                moves.append(action)
        return moves
