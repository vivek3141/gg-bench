import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 8 movement directions * 6 battle options (no battle + battle numbers 1-5)
        self.action_space = spaces.Discrete(48)

        # Observation space: 5x5 grid with values -1 (opponent), 0 (empty), 1 (current player)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Initialize units at their bases
        self.player_positions = {
            1: (0, 0),  # Player 1's unit at (0, 0)
            -1: (4, 4),  # Player -1 (Player 2)'s unit at (4, 4)
        }
        self.grid[0, 0] = 1
        self.grid[4, 4] = -1

        return np.array(self.grid, copy=True), {}

    def step(self, action):
        if self.done:
            return np.array(self.grid, copy=True), 0, True, False, {}

        # Parse action into movement and battle decision
        movement_action = action // 6  # 0 to 7
        battle_option = action % 6  # 0 (no battle) to 5 (battle number 5)

        # Get current player position
        x, y = self.player_positions[self.current_player]

        # Movement deltas corresponding to movement_action (directions)
        movement_deltas = [
            (-1, 0),  # Up
            (1, 0),  # Down
            (0, -1),  # Left
            (0, 1),  # Right
            (-1, -1),  # Up-Left
            (-1, 1),  # Up-Right
            (1, -1),  # Down-Left
            (1, 1),  # Down-Right
        ]

        dx, dy = movement_deltas[movement_action]
        x_new, y_new = x + dx, y + dy

        # Check if move is within bounds
        if not (0 <= x_new < 5 and 0 <= y_new < 5):
            # Invalid move
            self.done = True
            return np.array(self.grid, copy=True), -10, True, False, {}

        # Check if move is to adjacent cell
        if max(abs(dx), abs(dy)) > 1:
            self.done = True
            return np.array(self.grid, copy=True), -10, True, False, {}

        # Update grid and player position
        self.grid[x, y] = 0  # Remove unit from current position
        self.grid[x_new, y_new] = self.current_player
        self.player_positions[self.current_player] = (x_new, y_new)

        # Check if current player has reached opponent's base
        opponent_base = (4, 4) if self.current_player == 1 else (0, 0)
        if (x_new, y_new) == opponent_base:
            self.done = True
            return np.array(self.grid, copy=True), 1, True, False, {}

        # Check if opponent's unit is adjacent
        opponent_player = -self.current_player
        opp_x, opp_y = self.player_positions[opponent_player]
        if max(abs(x_new - opp_x), abs(y_new - opp_y)) <= 1:
            # Units are adjacent
            if battle_option == 0:
                # No battle chosen
                pass
            else:
                # Battle initiated
                player_number = battle_option  # 1 to 5
                opponent_number = self.np_random.integers(
                    1, 6
                )  # Random number between 1 and 5

                if player_number > opponent_number:
                    # Current player wins the battle
                    # Send opponent's unit back to their base
                    self.grid[opp_x, opp_y] = 0
                    if opponent_player == 1:
                        opp_base = (0, 0)
                    else:
                        opp_base = (4, 4)
                    self.grid[opp_base] = opponent_player
                    self.player_positions[opponent_player] = opp_base
                elif player_number == opponent_number:
                    # Tie, units remain in their positions
                    pass
                else:
                    # Opponent wins (nothing happens per the rules)
                    pass
        else:
            # Not adjacent; battle option should be zero
            if battle_option != 0:
                # Invalid action: attempting to battle when not adjacent
                self.done = True
                return np.array(self.grid, copy=True), -10, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        return np.array(self.grid, copy=True), 0, self.done, False, {}

    def render(self):
        board_str = ""
        symbols = {
            0: ".",
            1: "1",
            -1: "2",
        }
        for i in range(5):
            for j in range(5):
                board_str += symbols[self.grid[i, j]] + " "
            board_str += "\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        x, y = self.player_positions[self.current_player]

        movement_deltas = [
            (-1, 0),  # Up
            (1, 0),  # Down
            (0, -1),  # Left
            (0, 1),  # Right
            (-1, -1),  # Up-Left
            (-1, 1),  # Up-Right
            (1, -1),  # Down-Left
            (1, 1),  # Down-Right
        ]

        opponent_player = -self.current_player
        opp_x, opp_y = self.player_positions[opponent_player]

        for movement_action, (dx, dy) in enumerate(movement_deltas):
            x_new, y_new = x + dx, y + dy

            # Check if move is within bounds
            if not (0 <= x_new < 5 and 0 <= y_new < 5):
                continue  # Invalid move

            # Movement is valid
            # Check if opponent's unit would be adjacent after moving
            adj = max(abs(x_new - opp_x), abs(y_new - opp_y)) <= 1

            if adj:
                # Battle options: battle number 1-5 and no battle
                for battle_option in range(6):  # 0 (no battle) to 5
                    action_id = movement_action * 6 + battle_option
                    valid_actions.append(action_id)
            else:
                # Battle option must be 0 (no battle)
                action_id = movement_action * 6 + 0
                valid_actions.append(action_id)

        return valid_actions
