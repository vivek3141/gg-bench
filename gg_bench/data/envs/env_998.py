import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action space and observation space

        # Action space: Discrete(36), actions are indices from 0 to 35
        self.action_space = spaces.Discrete(36)

        # Observation space: Box(-4, 4, shape=(9,), dtype=np.int8)
        # Board positions are encoded using integers:
        # Red disks: R1=1, R2=2, R3=3, R4=4
        # Blue disks: B1=-1, B2=-2, B3=-3, B4=-4
        # Empty position: 0
        self.observation_space = spaces.Box(low=-4, high=4, shape=(9,), dtype=np.int8)

        # Mapping of disks to their integer values
        self.disk_values = {
            "R1": 1,
            "R2": 2,
            "R3": 3,
            "R4": 4,
            "B1": -1,
            "B2": -2,
            "B3": -3,
            "B4": -4,
        }

        # Define action mapping from index to (disk, target_position)
        self.actions_list = self._define_actions()

        # Initialize the environment
        self.reset()

    def _define_actions(self):
        """
        Returns a list mapping action indices to (disk, target_position).
        Each action corresponds to moving a specific disk to a specific position.
        """
        actions = []
        # Red disks
        red_disks = ["R1", "R2", "R3", "R4"]
        red_positions = {"R1": 1, "R2": 3, "R3": 5, "R4": 7}
        for disk in red_disks:
            start_pos = red_positions[disk]
            for pos in range(start_pos + 1, 10):  # Positions ahead
                actions.append((disk, pos))

        # Blue disks
        blue_disks = ["B1", "B2", "B3", "B4"]
        blue_positions = {"B1": 2, "B2": 4, "B3": 6, "B4": 8}
        for disk in blue_disks:
            start_pos = blue_positions[disk]
            for pos in range(start_pos + 1, 10):  # Positions ahead
                actions.append((disk, pos))
        return actions

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        Returns an observation and an info dictionary.
        """
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)
        # Initialize disk positions
        # Red disks
        self.board[0] = 1  # R1 at position 1 (index 0)
        self.board[2] = 2  # R2 at position 3 (index 2)
        self.board[4] = 3  # R3 at position 5 (index 4)
        self.board[6] = 4  # R4 at position 7 (index 6)
        # Blue disks
        self.board[1] = -1  # B1 at position 2 (index 1)
        self.board[3] = -2  # B2 at position 4 (index 3)
        self.board[5] = -3  # B3 at position 6 (index 5)
        self.board[7] = -4  # B4 at position 8 (index 7)
        self.current_player = 1  # 1 for Red, -1 for Blue
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        """
        Takes a step in the environment using the given action.
        Returns observation, reward, terminated, truncated, and info.
        """
        if self.done:
            return self.board.copy(), 0, True, False, {}
        # Map action index to move
        if action < 0 or action >= len(self.actions_list):
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {"error": "Invalid action index"},
            )
        disk, target_pos = self.actions_list[action]

        # Check if it's the current player's disk
        disk_value = self.disk_values[disk]
        if (self.current_player == 1 and disk_value < 0) or (
            self.current_player == -1 and disk_value > 0
        ):
            # Not current player's disk
            return self.board.copy(), -10, True, False, {"error": "Not your disk"}

        # Find the current position of the disk
        disk_pos = np.where(self.board == disk_value)[0]
        if len(disk_pos) == 0:
            return self.board.copy(), -10, True, False, {"error": "Disk not on board"}
        disk_pos = disk_pos[0]  # Current position index (0-based)

        target_index = target_pos - 1  # Convert position to index
        # Check if target position is ahead of disk's current position
        if target_index <= disk_pos:
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {"error": "Cannot move backward"},
            )
        # Check if target position is empty
        if self.board[target_index] != 0:
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {"error": "Target position not empty"},
            )
        # Move the disk
        self.board[disk_pos] = 0
        self.board[target_index] = disk_value

        # Check for victory
        if self._check_victory():
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch to next player
        self.current_player *= -1

        # Check if opponent has valid moves
        if not self.valid_moves():
            # Opponent has no valid moves, current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}

        return self.board.copy(), 0, False, False, {}

    def _check_victory(self):
        """
        Checks if the current player has won the game.
        Returns True if the current player wins, False otherwise.
        """
        # Get positions of current player's disks
        player_disk_values = (
            [1, 2, 3, 4] if self.current_player == 1 else [-1, -2, -3, -4]
        )
        opponent_disk_values = (
            [-1, -2, -3, -4] if self.current_player == 1 else [1, 2, 3, 4]
        )

        player_positions = [
            i for i, val in enumerate(self.board) if val in player_disk_values
        ]
        opponent_positions = [
            i for i, val in enumerate(self.board) if val in opponent_disk_values
        ]

        if not opponent_positions:
            # Opponent has no disks (should not happen but handle just in case)
            return True
        if not player_positions:
            # Current player has no disks (should not happen)
            return False

        # Lowest position of player's disks (indices are from 0)
        lowest_player_pos = min(player_positions)
        highest_opponent_pos = max(opponent_positions)

        # Check if all player's disks are ahead of opponent's disks
        if lowest_player_pos > highest_opponent_pos:
            return True
        return False

    def valid_moves(self):
        """
        Returns a list of valid action indices for the current player.
        """
        valid_actions = []
        player_disk_values = (
            [1, 2, 3, 4] if self.current_player == 1 else [-1, -2, -3, -4]
        )
        for idx, (disk, target_pos) in enumerate(self.actions_list):
            disk_value = self.disk_values[disk]
            if disk_value not in player_disk_values:
                continue  # Not current player's disk
            # Find the current position of the disk
            disk_pos = np.where(self.board == disk_value)[0]
            if len(disk_pos) == 0:
                continue  # Disk not on board
            disk_pos = disk_pos[0]
            target_index = target_pos - 1
            if target_index <= disk_pos:
                continue  # Cannot move backward
            if self.board[target_index] != 0:
                continue  # Target position not empty
            valid_actions.append(idx)
        return valid_actions

    def render(self):
        """
        Returns a string representation of the current board state.
        """
        board_repr = ""
        for i in range(9):
            val = self.board[i]
            if val == 0:
                board_repr += "[  ] "
            elif val > 0:
                disk = "R" + str(val)
                board_repr += "[" + disk + "] "
            else:
                disk = "B" + str(-val)
                board_repr += "[" + disk + "] "
        return board_repr.strip()
