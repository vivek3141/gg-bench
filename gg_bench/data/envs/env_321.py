import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # 0: Pass
        # 1-9: Influence node 0-8
        # 10-18: Collapse node 0-8
        self.action_space = spaces.Discrete(19)

        # Observation space:
        # Indices 0-8: Board state (values 0-4)
        # Index 9: Player Red's EP (0-5)
        # Index 10: Player Blue's EP (0-5)
        # Index 11: Current player (1 or 2)
        self.observation_space = spaces.Box(low=0, high=5, shape=(12,), dtype=np.int32)

        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game state
        self.board = np.zeros(
            9, dtype=np.int32
        )  # Board positions initialized to superposition (0)
        self.ep = {1: 5, 2: 5}  # Energy points for Player Red (1) and Player Blue (2)
        self.current_player = 1  # Player Red starts
        self.done = False

        return self.get_observation(), {}

    def step(self, action):
        # Replenish EP at the start of the turn
        self.ep[self.current_player] = min(5, self.ep[self.current_player] + 2)

        # Check if game is already over
        if self.done:
            return self.get_observation(), -10, True, False, {}

        # Validate the action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self.get_observation(), -10, True, False, {}

        reward = 0  # Default reward

        # Process the action
        if action == 0:  # Pass
            pass  # No EP cost
        elif 1 <= action <= 9:  # Influence a node
            node = action - 1
            self.ep[self.current_player] -= 1  # Cost 1 EP

            if self.board[node] == 0:
                self.board[node] = self.current_player  # Place influence marker
            elif self.board[node] == (3 - self.current_player):  # Opponent's marker
                self.board[node] = 0  # Remove both markers (cancel out)
        elif 10 <= action <= 18:  # Collapse a node
            node = action - 10
            self.ep[self.current_player] -= 2  # Cost 2 EP

            if self.board[node] == self.current_player:
                self.board[node] = self.current_player + 2  # Collapse to player's color
            else:
                self.done = True
                return self.get_observation(), -10, True, False, {}

        # Check for win condition
        if self.check_win_condition():
            self.done = True
            reward = 1
            return self.get_observation(), reward, True, False, {}

        # Switch to the opponent
        self.current_player = 3 - self.current_player

        # Check if opponent has any valid moves
        if not self.valid_moves():
            # Opponent cannot move, current player wins
            self.current_player = 3 - self.current_player  # Switch back to winner
            self.done = True
            reward = 1
            return self.get_observation(), reward, True, False, {}

        return self.get_observation(), reward, False, False, {}

    def render(self):
        board_str = "   A     B     C\n"
        for i in range(3):
            row_str = f"{i+1} "
            for j in range(3):
                idx = i * 3 + j
                state = self.board[idx]
                if state == 0:
                    cell = "[S]     "
                elif state == 1:
                    cell = "[S](R)  "
                elif state == 2:
                    cell = "[S](B)  "
                elif state == 3:
                    cell = "[R]     "
                elif state == 4:
                    cell = "[B]     "
                else:
                    cell = "Error   "
                row_str += cell
            board_str += row_str + "\n"
        ep_str = f"Player Red EP: {self.ep[1]}, Player Blue EP: {self.ep[2]}\n"
        current_player_str = (
            f"Current Player: {'Red' if self.current_player == 1 else 'Blue'}\n"
        )
        return board_str + ep_str + current_player_str

    def valid_moves(self):
        valid_actions = []

        # Pass is always valid
        valid_actions.append(0)

        ep = self.ep[self.current_player]

        # Influence actions
        if ep >= 1:
            for node in range(9):
                state = self.board[node]
                if state == 0 or state == (
                    3 - self.current_player
                ):  # Can influence or cancel opponent's influence
                    valid_actions.append(node + 1)

        # Collapse actions
        if ep >= 2:
            for node in range(9):
                if self.board[node] == self.current_player:  # Have own influence marker
                    valid_actions.append(node + 10)

        return valid_actions

    def get_observation(self):
        obs = np.zeros(12, dtype=np.int32)
        obs[:9] = self.board  # Board state
        obs[9] = self.ep[1]  # Player Red's EP
        obs[10] = self.ep[2]  # Player Blue's EP
        obs[11] = self.current_player  # Current player
        return obs

    def check_win_condition(self):
        player_collapsed_state = self.current_player + 2  # 3 for Red, 4 for Blue

        # Win by collapsing all nodes
        if np.all(self.board == player_collapsed_state):
            return True

        # Check if opponent has any moves
        opponent = 3 - self.current_player
        potential_ep = min(
            5, self.ep[opponent] + 2
        )  # EP replenished at start of their turn

        opponent_has_moves = False

        # Check influence actions
        if potential_ep >= 1:
            for node in range(9):
                state = self.board[node]
                if state == 0 or state == self.current_player:
                    opponent_has_moves = True
                    break

        # Check collapse actions
        if not opponent_has_moves and potential_ep >= 2:
            for node in range(9):
                if self.board[node] == opponent:
                    opponent_has_moves = True
                    break

        if not opponent_has_moves:
            return True  # Opponent cannot move, current player wins

        return False
