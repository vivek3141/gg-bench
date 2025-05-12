import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions:
        # 0: Move Forward (during movement phase)
        # 1: Move Backward (during movement phase)
        # 2-26: Battle choices (combinations of numbers 1-5 for both players)
        self.action_space = spaces.Discrete(27)

        # Observation space:
        # [cell1, cell2, cell3, cell4, cell5, current_player, phase]
        # cell values: 0 = empty, 1 = P1, 2 = P2
        # current_player: 1 or 2
        # phase: 0 = movement phase, 1 = battle phase
        self.observation_space = spaces.Box(low=0, high=5, shape=(7,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros(5, dtype=np.int32)
        self.grid[0] = 1  # P1 starts at cell 1 (index 0)
        self.grid[4] = 2  # P2 starts at cell 5 (index 4)
        self.current_player = 1
        self.phase = 0  # 0 = movement phase, 1 = battle phase
        self.done = False
        observation = self._get_obs()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        reward = 0
        info = {}

        if self.phase == 0:  # Movement phase
            if action not in [0, 1]:
                self.done = True
                reward = -10  # Invalid action
                return self._get_obs(), reward, True, False, info

            moved = self._move_player(action)
            if not moved:
                # Invalid move (attempted to move beyond grid)
                self.done = True
                reward = -10  # Invalid move
                return self._get_obs(), reward, True, False, info

            # Check for battle
            p1_pos = np.where(self.grid == 1)[0]
            p2_pos = np.where(self.grid == 2)[0]
            if len(p1_pos) > 0 and len(p2_pos) > 0 and p1_pos[0] == p2_pos[0]:
                # Both players on the same cell
                self.phase = 1  # Enter battle phase
            else:
                # Check for flag capture
                if self._check_win():
                    self.done = True
                    reward = 1  # Current player wins
                    return self._get_obs(), reward, True, False, info
                else:
                    # Switch to next player
                    self.current_player = 2 if self.current_player == 1 else 1
        else:  # Battle phase
            if action < 2 or action > 26:
                self.done = True
                reward = -10  # Invalid action
                return self._get_obs(), reward, True, False, info

            # Map action to battle choices
            index = action - 2
            p1_choice = index // 5 + 1
            p2_choice = index % 5 + 1

            # Resolve battle
            p1_pos = np.where(self.grid == 1)[0][0]
            p2_pos = np.where(self.grid == 2)[0][0]
            battle_pos = p1_pos  # Both are on the same cell

            if p1_choice > p2_choice:
                # P1 wins the battle
                if self.current_player == 1 and self._check_win():
                    # P1 captures flag
                    self.done = True
                    reward = 1
                    return self._get_obs(), reward, True, False, info
                self.grid[p2_pos] = 0  # Remove P2 from the cell
                self.grid[4] = 2  # P2 sent back to base
            elif p2_choice > p1_choice:
                # P2 wins the battle
                if self.current_player == 2 and self._check_win():
                    # P2 captures flag
                    self.done = True
                    reward = 1
                    return self._get_obs(), reward, True, False, info
                self.grid[p1_pos] = 0  # Remove P1 from the cell
                self.grid[0] = 1  # P1 sent back to base
            else:
                # Tie, both go back to base
                self.grid[battle_pos] = 0
                self.grid[0] = 1
                self.grid[4] = 2

            self.phase = 0  # Back to movement phase
            self.current_player = 2 if self.current_player == 1 else 1

        observation = self._get_obs()
        return observation, reward, self.done, False, info

    def _move_player(self, action):
        player = self.current_player
        player_pos = np.where(self.grid == player)[0][0]
        if action == 0:
            # Move forward
            if player == 1:
                new_pos = player_pos + 1
            else:
                new_pos = player_pos - 1
        else:
            # Move backward
            if player == 1:
                new_pos = player_pos - 1
            else:
                new_pos = player_pos + 1

        # Check boundaries
        if new_pos < 0 or new_pos > 4:
            return False  # Invalid move

        # Move player
        self.grid[player_pos] = 0
        self.grid[new_pos] = player
        return True

    def _check_win(self):
        player = self.current_player
        opponent = 2 if player == 1 else 1
        player_pos = np.where(self.grid == player)[0][0]
        opponent_base = 4 if opponent == 2 else 0
        opponent_pos = np.where(self.grid == opponent)[0][0]
        if player_pos == opponent_base and opponent_pos != opponent_base:
            return True  # Current player wins
        return False

    def render(self):
        grid_str = ""
        for idx in range(5):
            cell = self.grid[idx]
            if cell == 1:
                grid_str += "[P1]"
            elif cell == 2:
                grid_str += "[P2]"
            else:
                grid_str += "[ ]"
        phase_str = "Movement Phase" if self.phase == 0 else "Battle Phase"
        grid_str += f"  {phase_str}, Current Player: P{self.current_player}"
        return grid_str

    def valid_moves(self):
        if self.done:
            return []

        if self.phase == 0:
            return [0, 1]  # Move forward or backward
        else:
            return list(range(2, 27))  # Battle choices

    def _get_obs(self):
        # Observation: [cell1, cell2, cell3, cell4, cell5, current_player, phase]
        obs = np.zeros(7, dtype=np.int32)
        obs[0:5] = self.grid
        obs[5] = self.current_player
        obs[6] = self.phase
        return obs
