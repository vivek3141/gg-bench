import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions:
        # 0-8: Observe particle at position 0-8
        # 9: Collapse entangled opponent particles
        # 10-18: Attempt Quantum Swap at position 0-8
        # 19-27: Attempt Quantum Tunnel at position 0-8

        self.action_space = spaces.Discrete(28)

        # Observation space is a grid of size 9 with values from 0 to 4
        # 0: Superposed (S)
        # 1: Claimed by Player 1 (P1)
        # 2: Claimed by Player 2 (P2)
        # 3: Entangled opponent's particle
        # 4: Neutral state (X) after collapse

        self.observation_space = spaces.Box(low=0, high=4, shape=(9,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros(9, dtype=np.int8)  # All particles start as Superposed (0)
        self.current_player = 1  # 1 for Player 1, 2 for Player 2
        self.done = False
        self.entangled_positions = set()  # Positions of entangled opponent particles
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), -10, True, False, {}

        reward = 0

        if action < 0 or action >= 28:
            return self.grid.copy(), -10, True, False, {}  # Invalid action

        if action <= 8:
            # Observe particle at position action
            pos = action
            if self.grid[pos] != 0:
                # Invalid move
                self.done = True
                return self.grid.copy(), -10, True, False, {}
            else:
                # Observe the particle
                self.grid[pos] = self.current_player
                # Check for adjacency to opponent's particles to entangle them
                self.entangle_adjacent_opponent_particles(pos)

        elif action == 9:
            # Attempt to collapse entangled opponent particles
            if len(self.entangled_positions) == 0:
                # Invalid move
                self.done = True
                return self.grid.copy(), -10, True, False, {}
            else:
                # Collapse entangled opponent particles
                self.collapse_entangled_particles()

        elif 10 <= action <= 18:
            # Attempt Quantum Swap at position action - 10
            pos = action - 10
            if not self.attempt_quantum_swap(pos):
                # Invalid move
                self.done = True
                return self.grid.copy(), -10, True, False, {}

        elif 19 <= action <= 27:
            # Attempt Quantum Tunnel at position action - 19
            pos = action - 19
            if not self.attempt_quantum_tunnel(pos):
                # Invalid move
                self.done = True
                return self.grid.copy(), -10, True, False, {}

        else:
            # Invalid action
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        # After processing action, check win condition
        if self.check_win_condition():
            reward = 1
            self.done = True
        else:
            # Switch player
            self.current_player = 1 if self.current_player == 2 else 2

        return self.grid.copy(), reward, self.done, False, {}

    def entangle_adjacent_opponent_particles(self, pos):
        # Get positions adjacent (up, down, left, right) to pos
        adjacent_positions = self.get_adjacent_positions(pos)
        opponent = 1 if self.current_player == 2 else 2
        for adj_pos in adjacent_positions:
            if self.grid[adj_pos] == opponent:
                # Mark as entangled
                self.grid[adj_pos] = 3  # 3 represents entangled opponent's particle
                self.entangled_positions.add(adj_pos)

    def get_adjacent_positions(self, pos):
        row = pos // 3
        col = pos % 3
        adjacent_positions = []
        # Up
        if row > 0:
            adjacent_positions.append((row - 1) * 3 + col)
        # Down
        if row < 2:
            adjacent_positions.append((row + 1) * 3 + col)
        # Left
        if col > 0:
            adjacent_positions.append(row * 3 + (col - 1))
        # Right
        if col < 2:
            adjacent_positions.append(row * 3 + (col + 1))
        return adjacent_positions

    def collapse_entangled_particles(self):
        for pos in self.entangled_positions:
            self.grid[pos] = 4  # Set to neutral state (X)
        self.entangled_positions.clear()

    def attempt_quantum_swap(self, pos):
        opponent = 1 if self.current_player == 2 else 2
        if self.grid[pos] != opponent:
            return False  # Invalid action, position does not have opponent's particle
        # Check for exactly two of our claimed particles adjacent to this position
        adjacent_positions = self.get_adjacent_positions(pos)
        our_claimed_positions = [
            adj_pos
            for adj_pos in adjacent_positions
            if self.grid[adj_pos] == self.current_player
        ]
        if len(our_claimed_positions) == 2:
            # Swap one of our adjacent particles with opponent's particle at pos
            swap_pos = our_claimed_positions[0]  # Swap with the first found
            self.grid[swap_pos], self.grid[pos] = self.grid[pos], self.grid[swap_pos]
            return True
        else:
            return False  # Conditions not met

    def attempt_quantum_tunnel(self, pos):
        if self.grid[pos] != 0:
            return False  # Invalid action, position is not Superposed
        if self.find_positions_for_tunnel(pos):
            # Claim the Superposed particle at pos
            self.grid[pos] = self.current_player
            return True
        else:
            return False  # Conditions not met

    def find_positions_for_tunnel(self, pos):
        # Possible lines to check
        lines = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],  # Rows
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],  # Columns
            [0, 4, 8],
            [2, 4, 6],  # Diagonals
        ]
        for line in lines:
            if pos in line:
                other_positions = [p for p in line if p != pos]
                if (
                    self.grid[other_positions[0]] == self.current_player
                    and self.grid[other_positions[1]] == self.current_player
                ):
                    return True
        return False

    def check_win_condition(self):
        opponent = 1 if self.current_player == 2 else 2
        opponent_claimed_positions = np.where(self.grid == opponent)[0]
        return len(opponent_claimed_positions) == 0

    def valid_moves(self):
        valid_actions = []
        if self.done:
            return valid_actions
        # Actions 0-8: Observe particle at position 0-8
        for pos in range(9):
            if self.grid[pos] == 0:
                valid_actions.append(pos)
        # Action 9: Collapse entangled opponent particles
        if len(self.entangled_positions) > 0:
            valid_actions.append(9)
        # Actions 10-18: Attempt Quantum Swap at position 0-8
        for pos in range(9):
            if self.can_attempt_quantum_swap(pos):
                valid_actions.append(10 + pos)
        # Actions 19-27: Attempt Quantum Tunnel at position 0-8
        for pos in range(9):
            if self.can_attempt_quantum_tunnel(pos):
                valid_actions.append(19 + pos)
        return valid_actions

    def can_attempt_quantum_swap(self, pos):
        opponent = 1 if self.current_player == 2 else 2
        if self.grid[pos] != opponent:
            return False
        adjacent_positions = self.get_adjacent_positions(pos)
        our_claimed_positions = [
            adj_pos
            for adj_pos in adjacent_positions
            if self.grid[adj_pos] == self.current_player
        ]
        return len(our_claimed_positions) == 2

    def can_attempt_quantum_tunnel(self, pos):
        if self.grid[pos] != 0:
            return False
        return self.find_positions_for_tunnel(pos)

    def render(self):
        grid_str = ""
        for i in range(3):
            grid_str += "|"
            for j in range(3):
                pos = i * 3 + j
                if self.grid[pos] == 0:
                    grid_str += " S |"
                elif self.grid[pos] == 1:
                    grid_str += "P1 |"
                elif self.grid[pos] == 2:
                    grid_str += "P2 |"
                elif self.grid[pos] == 3:
                    grid_str += " E |"  # 'E' for entangled particle
                elif self.grid[pos] == 4:
                    grid_str += " X |"  # Neutral state
            grid_str += "\n"
        return grid_str
