import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(9) for 9 cells to attack
        self.action_space = spaces.Discrete(9)

        # Observation space: For each player (2 players), 9 cells, 2 values per cell (attack status and clue)
        # Attack status: 0 (Unattacked), 1 (Miss), 2 (Hit)
        # Clue: 0 (No clue), 1-6 as per clue codes
        self.observation_space = spaces.Box(
            low=0, high=6, shape=(2, 9, 2), dtype=np.int8
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly place crowns for both players
        self.crown_positions = {
            1: self.np_random.integers(0, 9),
            -1: self.np_random.integers(0, 9),
        }

        # Initialize attack records and clues for both players
        # For each player: array of shape (9, 2)
        # attack_status: 0 (Unattacked), 1 (Miss), 2 (Hit)
        # clue_code: 0 (No clue), 1-6 as per clue codes
        self.player_data = {
            1: np.zeros((9, 2), dtype=np.int8),
            -1: np.zeros((9, 2), dtype=np.int8),
        }

        # Keep track of opponent's attacks on each player's own grid
        self.opponent_attacks = {
            1: np.zeros(9, dtype=np.int8),
            -1: np.zeros(9, dtype=np.int8),
        }
        self.done = False
        self.current_player = 1  # Player 1 starts
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if action is valid
        if action < 0 or action >= 9:
            # Invalid action index
            return self._get_observation(), -10, True, False, {}
        if self.player_data[self.current_player][action][0] != 0:
            # Cell has already been attacked
            return self._get_observation(), -10, True, False, {}

        # Process the attack
        opponent = -self.current_player
        opponent_crown = self.crown_positions[opponent]

        if action == opponent_crown:
            # Hit
            self.player_data[self.current_player][action][0] = 2  # Hit
            self.done = True
            return self._get_observation(), 1, True, False, {}
        else:
            # Miss
            self.player_data[self.current_player][action][0] = 1  # Miss

            # Provide a clue
            clue_type, clue_value = self._provide_clue(action, opponent_crown)
            clue_code = self._encode_clue(clue_type, clue_value)
            self.player_data[self.current_player][action][1] = clue_code

            # Record the opponent's attack on current player's grid
            # Opponent's move (we assume the agent plays both sides)
            opponent_action = self._opponent_policy(self.player_data[opponent])
            if opponent_action is None:
                # No valid moves left for opponent
                self.done = True
                return self._get_observation(), 0, True, False, {}

            # Check if opponent's attack is a Hit
            own_crown = self.crown_positions[self.current_player]
            if opponent_action == own_crown:
                # Opponent hits current player's crown
                self.player_data[opponent][opponent_action][0] = 2  # Hit
                self.done = True
                # As per game rules, current player loses
                return self._get_observation(), -1, True, False, {}
            else:
                # Opponent misses
                self.player_data[opponent][opponent_action][0] = 1  # Miss
                # Provide clue to opponent (update their data)
                clue_type_opponent, clue_value_opponent = self._provide_clue(
                    opponent_action, own_crown
                )
                clue_code_opponent = self._encode_clue(
                    clue_type_opponent, clue_value_opponent
                )
                self.player_data[opponent][opponent_action][1] = clue_code_opponent

            self.current_player *= -1  # Switch player
            return self._get_observation(), 0, False, False, {}

    def render(self):
        board_str = f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        for player in [1, -1]:
            board_str += f"Player {1 if player == 1 else 2} attack grid:\n"
            for i in range(3):
                row_str = ""
                for j in range(3):
                    idx = i * 3 + j
                    status = self.player_data[player][idx][0]
                    clue_code = self.player_data[player][idx][1]
                    if status == 0:
                        cell_str = " "
                    elif status == 1:
                        cell_str = "M"  # Miss
                    elif status == 2:
                        cell_str = "H"  # Hit
                    else:
                        cell_str = "?"

                    if clue_code != 0:
                        cell_str += self._decode_clue(clue_code)
                    else:
                        cell_str += " "

                    row_str += f"| {cell_str} "
                row_str += "|\n"
                board_str += row_str
            board_str += "\n"
        return board_str

    def valid_moves(self):
        # Return indices of un-attacked cells for current_player
        return [i for i in range(9) if self.player_data[self.current_player][i][0] == 0]

    def _get_observation(self):
        # Combine both players' data into observation
        obs = np.stack(
            (
                self.player_data[1].flatten(),
                self.player_data[-1].flatten(),
            ),
            axis=0,
        )
        obs = obs.reshape(-1)
        return obs

    def _provide_clue(self, action_idx, crown_idx):
        action_row, action_col = divmod(action_idx, 3)
        crown_row, crown_col = divmod(crown_idx, 3)

        # Randomly choose to provide a row or column hint
        if self.np_random.random() < 0.5:
            clue_type = "row"
            if crown_row > action_row:
                clue_value = 1  # Greater than
            elif crown_row == action_row:
                clue_value = 0  # Equal to
            else:
                clue_value = -1  # Less than
        else:
            clue_type = "col"
            if crown_col > action_col:
                clue_value = 1  # Greater than
            elif crown_col == action_col:
                clue_value = 0  # Equal to
            else:
                clue_value = -1  # Less than
        return clue_type, clue_value

    def _encode_clue(self, clue_type, clue_value):
        # Clue codes:
        # 0: No clue
        # 1: Row less than (-1)
        # 2: Row equal to (0)
        # 3: Row greater than (1)
        # 4: Column less than (-1)
        # 5: Column equal to (0)
        # 6: Column greater than (1)
        if clue_type == "row":
            if clue_value == -1:
                return 1
            elif clue_value == 0:
                return 2
            elif clue_value == 1:
                return 3
        elif clue_type == "col":
            if clue_value == -1:
                return 4
            elif clue_value == 0:
                return 5
            elif clue_value == 1:
                return 6
        return 0  # Should not reach here

    def _decode_clue(self, clue_code):
        if clue_code == 1:
            return "R<"
        elif clue_code == 2:
            return "R="
        elif clue_code == 3:
            return "R>"
        elif clue_code == 4:
            return "C<"
        elif clue_code == 5:
            return "C="
        elif clue_code == 6:
            return "C>"
        else:
            return "  "

    def _opponent_policy(self, opponent_data):
        # Simple policy: choose a random valid move
        valid_moves = [i for i in range(9) if opponent_data[i][0] == 0]
        if not valid_moves:
            return None
        return self.np_random.choice(valid_moves)
