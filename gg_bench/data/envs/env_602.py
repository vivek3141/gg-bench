import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0-2 are move forward 1-3 steps
        # Actions: 3-23 are place obstacle at cell (action - 3)
        self.action_space = spaces.Discrete(24)

        # Observation space: 21 cells with values:
        # 0: Empty cell
        # 1: Current player's token
        # -1: Opponent's token
        # 2: Current player's obstacle
        # -2: Opponent's obstacle
        self.observation_space = spaces.Box(low=-2, high=2, shape=(21,), dtype=np.int8)

        # Initialize the board and game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(21, dtype=np.int8)
        self.board[0] = 1  # Player 1 starts at cell 0
        self.board[20] = -1  # Player 2 starts at cell 20

        self.current_player = 1  # Player 1 starts
        self.done = False

        # Obstacles remaining for each player
        self.obstacles_remaining = {1: 5, -1: 5}

        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}  # Game already over

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {},
            )  # Invalid move, game over with penalty

        reward = 0

        if action <= 2:
            # Movement action
            steps = action + 1  # action 0 maps to 1 step, etc.
            current_pos = np.where(self.board == self.current_player)[0][0]
            direction = (
                1 if self.current_player == 1 else -1
            )  # Player 1 moves right, Player 2 moves left
            target_pos = current_pos + direction * steps

            # Check for obstacles or opponent's token in the path
            if direction == 1:
                path = self.board[current_pos + 1 : target_pos + 1]
            else:
                path = self.board[target_pos:current_pos]
            if (path == -self.current_player * 2).any() or (
                path == -self.current_player
            ).any():
                # Obstacle or opponent in the way
                self.done = True
                return (
                    self.board.copy(),
                    -10,
                    True,
                    False,
                    {},
                )  # Invalid move, game over with penalty

            # Move the token
            self.board[current_pos] = 0
            # Check if landing on opponent's obstacle
            if self.board[target_pos] == -2 * self.current_player:
                self.board[target_pos] = (
                    self.current_player
                )  # Remove obstacle, occupy cell
            else:
                self.board[target_pos] = self.current_player  # Occupy cell

            # Check for win condition
            if (self.current_player == 1 and target_pos == 20) or (
                self.current_player == -1 and target_pos == 0
            ):
                reward = 1  # Win reward
                self.done = True
                return self.board.copy(), reward, True, False, {}

        else:
            # Obstacle placement action
            place_pos = action - 3  # action 3 corresponds to cell 0
            if self.obstacles_remaining[self.current_player] <= 0:
                # No obstacles left
                self.done = True
                return (
                    self.board.copy(),
                    -10,
                    True,
                    False,
                    {},
                )  # Invalid move, game over with penalty

            # Check placement range
            player_pos = np.where(self.board == self.current_player)[0][0]
            direction = 1 if self.current_player == 1 else -1
            max_range = player_pos + direction * 5
            if direction == 1:
                allowed_range = range(player_pos + 1, min(max_range + 1, 21))
            else:
                allowed_range = range(max(max_range, -1), player_pos)

            if place_pos not in allowed_range:
                # Cannot place obstacle out of range
                self.done = True
                return (
                    self.board.copy(),
                    -10,
                    True,
                    False,
                    {},
                )  # Invalid move, game over with penalty

            if self.board[place_pos] != 0:
                # Cannot place obstacle on occupied cell
                self.done = True
                return (
                    self.board.copy(),
                    -10,
                    True,
                    False,
                    {},
                )  # Invalid move, game over with penalty

            # Place the obstacle
            self.board[place_pos] = 2 * self.current_player
            self.obstacles_remaining[self.current_player] -= 1

            # Check that path is not fully blocked
            if not self._is_path_open(-self.current_player):
                # Must not block the opponent completely
                # Undo the obstacle placement
                self.board[place_pos] = 0
                self.obstacles_remaining[self.current_player] += 1
                self.done = True
                return (
                    self.board.copy(),
                    -10,
                    True,
                    False,
                    {},
                )  # Invalid move, game over with penalty

        # Switch player
        self.current_player *= -1

        return self.board.copy(), reward, self.done, False, {}

    def _is_path_open(self, player):
        # Check if there is at least one route for the opponent to advance
        player_pos = np.where(self.board == player)[0][0]
        direction = 1 if player == 1 else -1
        target_cell = 20 if player == 1 else 0

        position = player_pos
        while 0 <= position <= 20:
            position += direction
            if not (0 <= position <= 20):
                break
            cell_value = self.board[position]
            if cell_value == 0 or cell_value == -2 * player:
                # Empty cell or can remove opponent's obstacle
                return True
            elif cell_value == 2 * player or cell_value == player:
                # Own obstacle or own token, skip
                continue
            else:
                # Opponent's token or obstacle, path blocked here
                break
        return False

    def render(self):
        board_str = ""
        for i in range(21):
            cell = self.board[i]
            if cell == 0:
                board_str += "[ ]"
            elif cell == 1:
                board_str += "[P1]"
            elif cell == -1:
                board_str += "[P2]"
            elif cell == 2:
                board_str += "[X1]"
            elif cell == -2:
                board_str += "[X2]"
        return board_str

    def valid_moves(self):
        valid_actions = []

        # Movement actions
        current_pos = np.where(self.board == self.current_player)[0][0]
        direction = 1 if self.current_player == 1 else -1

        for steps in [1, 2, 3]:
            target_pos = current_pos + direction * steps
            if not (0 <= target_pos <= 20):
                continue  # Can't move beyond the board
            # Check for obstacles or opponent in the path
            if direction == 1:
                path = self.board[current_pos + 1 : target_pos + 1]
            else:
                path = self.board[target_pos:current_pos]
            if (path == -self.current_player * 2).any() or (
                path == -self.current_player
            ).any():
                continue  # Obstacle or opponent's token in the way
            valid_actions.append(steps - 1)  # action indices 0-2

        # Obstacle placement actions
        if self.obstacles_remaining[self.current_player] > 0:
            player_pos = np.where(self.board == self.current_player)[0][0]
            direction = 1 if self.current_player == 1 else -1
            max_range = player_pos + direction * 5
            if direction == 1:
                allowed_range = range(player_pos + 1, min(max_range + 1, 21))
            else:
                allowed_range = range(max(max_range, -1), player_pos)
            for pos in allowed_range:
                if self.board[pos] == 0:
                    # Check that placing obstacle does not block opponent completely
                    self.board[pos] = (
                        2 * self.current_player
                    )  # Temporarily place obstacle
                    if self._is_path_open(-self.current_player):
                        valid_actions.append(pos + 3)  # action indices 3-23
                    self.board[pos] = 0  # Remove temporary obstacle
        return valid_actions
