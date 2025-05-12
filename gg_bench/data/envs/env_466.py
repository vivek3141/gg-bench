import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(50) - 5 shadows * 10 possible actions
        self.action_space = spaces.Discrete(50)

        # Observation space: (5, 5, 2)
        # Channel 0: 0 - empty, 1 - Player 1's shadow, 2 - Player 2's shadow
        # Channel 1: Energy level of the shadow (0-5)
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(5, 5, 2), dtype=np.int32
        )

        # Game constants
        self.grid_size = 5
        self.max_energy = 5
        self.min_energy = 0

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board
        self.board = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.int32)

        # Player shadows dictionary
        # Each player has a list of shadows, each shadow is a dict with position and energy
        self.player_shadows = {1: [], 2: []}

        # Initialize Player 1's shadows
        for y in range(self.grid_size):
            shadow = {"position": (y, 0), "energy": 3}  # (row, column)
            self.player_shadows[1].append(shadow)
            self.board[y, 0, 0] = 1  # Player 1's shadow
            self.board[y, 0, 1] = 3  # Energy

        # Initialize Player 2's shadows
        for y in range(self.grid_size):
            shadow = {"position": (y, self.grid_size - 1), "energy": 3}  # (row, column)
            self.player_shadows[2].append(shadow)
            self.board[y, self.grid_size - 1, 0] = 2  # Player 2's shadow
            self.board[y, self.grid_size - 1, 1] = 3  # Energy

        self.current_player = 1  # Player 1 starts
        self.done = False
        self.info = {}
        return self.board.copy(), self.info  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, self.info

        # Decode the action
        shadow_index = action // 10
        action_index = action % 10

        # Get the player's shadows
        shadows = self.player_shadows[self.current_player]

        if shadow_index >= len(shadows):
            # Invalid shadow index
            reward = -10
            self.done = False
            return self.board.copy(), reward, False, False, self.info

        shadow = shadows[shadow_index]
        shadow_pos = shadow["position"]
        shadow_energy = shadow["energy"]

        # Check if action is valid
        if action_index == 0:
            # Rest - do nothing
            pass
        elif action_index >= 1 and action_index <= 4:
            # Move action
            if shadow_energy < 1:
                # Not enough energy to move
                reward = -10
                self.done = False
                return self.board.copy(), reward, False, False, self.info

            # Determine new position
            new_pos = self._get_new_position(shadow_pos, action_index)
            if not self._is_valid_position(new_pos):
                # Invalid move
                reward = -10
                self.done = False
                return self.board.copy(), reward, False, False, self.info

            # Check if the cell is empty
            if self.board[new_pos[0], new_pos[1], 0] != 0:
                # Cell is occupied
                reward = -10
                self.done = False
                return self.board.copy(), reward, False, False, self.info

            # Move the shadow
            self.board[shadow_pos[0], shadow_pos[1], :] = 0  # Clear old position
            shadow["position"] = new_pos
            shadow["energy"] -= 1
            self.board[new_pos[0], new_pos[1], 0] = self.current_player
            self.board[new_pos[0], new_pos[1], 1] = shadow["energy"]
        elif action_index >= 5 and action_index <= 8:
            # Cast action
            if shadow_energy < 2:
                # Not enough energy to cast
                reward = -10
                self.done = False
                return self.board.copy(), reward, False, False, self.info

            # Determine target position
            target_pos = self._get_new_position(shadow_pos, action_index - 4)
            if not self._is_valid_position(target_pos):
                # Invalid cast
                reward = -10
                self.done = False
                return self.board.copy(), reward, False, False, self.info

            # Apply casting
            opponent = 2 if self.current_player == 1 else 1
            if self.board[target_pos[0], target_pos[1], 0] == opponent:
                # Opponent's shadow is hit
                shadow["energy"] -= 2
                self.board[shadow_pos[0], shadow_pos[1], 1] = shadow["energy"]
                # Reduce opponent's shadow energy
                for opp_shadow in self.player_shadows[opponent]:
                    if opp_shadow["position"] == target_pos:
                        opp_shadow["energy"] -= 1
                        self.board[target_pos[0], target_pos[1], 1] = opp_shadow[
                            "energy"
                        ]
                        if opp_shadow["energy"] <= 0:
                            # Opponent's shadow is captured
                            self.board[target_pos[0], target_pos[1], :] = 0
                            self.player_shadows[opponent].remove(opp_shadow)
                        break
            else:
                # Cast to empty or own shadow
                shadow["energy"] -= 2
                self.board[shadow_pos[0], shadow_pos[1], 1] = shadow["energy"]
        elif action_index == 9:
            # Overcharge Cast
            if shadow_energy < 5:
                # Not enough energy to overcharge
                reward = -10
                self.done = False
                return self.board.copy(), reward, False, False, self.info

            # Overcharge affects all adjacent cells
            affected_positions = self._get_adjacent_positions(shadow_pos)
            opponent = 2 if self.current_player == 1 else 1
            shadow["energy"] = 0  # Overcharge uses all energy
            self.board[shadow_pos[0], shadow_pos[1], 1] = shadow["energy"]
            # Shadow is captured after overcharge
            self.board[shadow_pos[0], shadow_pos[1], :] = 0
            shadows.remove(shadow)
            for pos in affected_positions:
                if self.board[pos[0], pos[1], 0] == opponent:
                    # Reduce opponent's shadow energy by 2
                    for opp_shadow in self.player_shadows[opponent]:
                        if opp_shadow["position"] == pos:
                            opp_shadow["energy"] -= 2
                            self.board[pos[0], pos[1], 1] = opp_shadow["energy"]
                            if opp_shadow["energy"] <= 0:
                                # Opponent's shadow is captured
                                self.board[pos[0], pos[1], :] = 0
                                self.player_shadows[opponent].remove(opp_shadow)
                            break
        else:
            # Invalid action
            reward = -10
            self.done = False
            return self.board.copy(), reward, False, False, self.info

        # Energy Phase: Shadows that did not move regain 1 energy
        # In this simplified version, only the shadow that performed the action is considered
        # So no shadows regain energy in this implementation

        # Check for win condition
        opponent = 2 if self.current_player == 1 else 1
        if len(self.player_shadows[opponent]) == 0:
            # Current player wins
            reward = 1
            self.done = True
            return self.board.copy(), reward, True, False, self.info

        # Switch to the other player
        self.current_player = opponent

        reward = 0  # No reward for a regular move
        return self.board.copy(), reward, False, False, self.info

    def render(self):
        board_str = ""
        for y in range(self.grid_size):
            row_str = ""
            for x in range(self.grid_size):
                cell_player = self.board[y, x, 0]
                cell_energy = self.board[y, x, 1]
                if cell_player == 1:
                    cell_str = f"S1({cell_energy})"
                elif cell_player == 2:
                    cell_str = f"S2({cell_energy})"
                else:
                    cell_str = "   "
                row_str += f"|{cell_str}"
            row_str += "|\n"
            board_str += row_str
        return board_str

    def valid_moves(self):
        valid_actions = []
        shadows = self.player_shadows[self.current_player]
        opponent = 2 if self.current_player == 1 else 1
        for idx, shadow in enumerate(shadows):
            shadow_pos = shadow["position"]
            shadow_energy = shadow["energy"]
            actions = []
            # Rest is always valid
            actions.append(idx * 10 + 0)
            # Move actions
            if shadow_energy >= 1:
                for action_index in range(1, 5):
                    new_pos = self._get_new_position(shadow_pos, action_index)
                    if (
                        self._is_valid_position(new_pos)
                        and self.board[new_pos[0], new_pos[1], 0] == 0
                    ):
                        actions.append(idx * 10 + action_index)
            # Cast actions
            if shadow_energy >= 2:
                for action_index in range(5, 9):
                    target_pos = self._get_new_position(shadow_pos, action_index - 4)
                    if self._is_valid_position(target_pos):
                        actions.append(idx * 10 + action_index)
            # Overcharge
            if shadow_energy == 5:
                actions.append(idx * 10 + 9)
            valid_actions.extend(actions)
        return valid_actions

    def _get_new_position(self, pos, direction):
        # Directions:
        # 1: Up (-1, 0)
        # 2: Down (+1, 0)
        # 3: Left (0, -1)
        # 4: Right (0, +1)
        row, col = pos
        if direction == 1:
            return (row - 1, col)
        elif direction == 2:
            return (row + 1, col)
        elif direction == 3:
            return (row, col - 1)
        elif direction == 4:
            return (row, col + 1)
        else:
            return pos

    def _is_valid_position(self, pos):
        row, col = pos
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def _get_adjacent_positions(self, pos):
        row, col = pos
        positions = []
        directions = [1, 2, 3, 4]
        for direction in directions:
            new_pos = self._get_new_position(pos, direction)
            if self._is_valid_position(new_pos):
                positions.append(new_pos)
        return positions
