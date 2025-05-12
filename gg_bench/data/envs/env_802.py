import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 18 possible actions: 0-8 for Occupy actions, 9-17 for Attack actions
        self.action_space = spaces.Discrete(18)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Initialize the adjacency list for each zone
        self.adjacency_list = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4, 6],
            4: [1, 3, 5, 7],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7],
        }

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # 1 for Player 1 ('X'), -1 for Player 2 ('O')
        self.done = False
        self.prev_attacked_zone = {1: None, -1: None}
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        if not self.is_valid_action(action):
            self.done = True
            return self.board.copy(), -10, True, False, {}

        reward = 0
        terminated = False

        # Process the action
        if action < 9:
            # Occupy action
            zone = action
            self.board[zone] = self.current_player
            # Reset previous attacked zone since occupancy does not count as an attack
            self.prev_attacked_zone[self.current_player] = None
        else:
            # Attack action
            zone = action - 9
            self.board[zone] = self.current_player
            self.prev_attacked_zone[self.current_player] = zone

        # Check for victory condition
        player_control_count = np.count_nonzero(self.board == self.current_player)
        if player_control_count >= 5:
            reward = 1
            terminated = True
            self.done = True

        # Switch to the other player
        self.current_player *= -1

        return self.board.copy(), reward, terminated, False, {}

    def render(self):
        symbols = {1: "X", -1: "O", 0: " "}
        board_str = "\nCurrent Grid:\n"
        board_str += "-------------\n"
        for i in range(3):
            board_str += "|"
            for j in range(3):
                zone = i * 3 + j
                board_str += f" {symbols[self.board[zone]]} |"
            board_str += "\n-------------\n"
        return board_str

    def valid_moves(self):
        valid_actions = []

        # Occupy actions
        for zone in range(9):
            if self.board[zone] == 0:
                valid_actions.append(zone)

        # Attack actions
        for zone in range(9):
            if self.can_attack(zone):
                valid_actions.append(zone + 9)

        return valid_actions

    def is_valid_action(self, action):
        if action < 0 or action >= 18:
            return False

        if action < 9:
            # Occupy action
            zone = action
            if self.board[zone] != 0:
                return False
            else:
                return True
        else:
            # Attack action
            zone = action - 9
            return self.can_attack(zone)

    def can_attack(self, zone):
        # Check if the zone is controlled by the opponent
        if self.board[zone] != -self.current_player:
            return False

        # Check if the player did not attack the same zone in their immediate previous turn
        if self.prev_attacked_zone[self.current_player] == zone:
            return False

        # Check if the zone is adjacent to at least one zone controlled by the player
        adjacent_zones = self.adjacency_list[zone]
        for adj_zone in adjacent_zones:
            if self.board[adj_zone] == self.current_player:
                return True

        return False
