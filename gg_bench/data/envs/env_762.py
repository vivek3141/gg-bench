import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(
            low=-1.0, high=100.0, shape=(28,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the grid and particles
        self.grid = np.zeros(25, dtype=np.float32)
        self.particles = []
        positions = np.random.choice(25, size=10, replace=True)
        for pos in positions:
            value = np.random.randint(1, 4)  # Particle value between 1 and 3
            self.particles.append({"position": pos, "value": value})
        # Update the grid with initial particle values
        for particle in self.particles:
            pos = particle["position"]
            self.grid[pos] += particle["value"]

        # Initialize player scores and current player
        self.scores = {1: 0, -1: 0}
        self.current_player = np.random.choice([1, -1])
        self.done = False

        # Return the initial observation and info
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            observation = self._get_observation()
            return observation, -10, True, False, {}

        if action < 0 or action >= 25:
            # Invalid action
            self.done = True
            observation = self._get_observation()
            return observation, -10, True, False, {}

        reward = 0

        # Player scans the selected coordinate
        particles_captured = [p for p in self.particles if p["position"] == action]
        if particles_captured:
            total_value = sum(p["value"] for p in particles_captured)
            self.scores[self.current_player] += total_value
            # Remove captured particles from the grid and particle list
            self.particles = [p for p in self.particles if p["position"] != action]
            self.grid[action] = 0
        else:
            # No particle found; nothing happens
            pass

        # Check for winning condition
        if self.scores[self.current_player] >= 10:
            self.done = True
            observation = self._get_observation()
            return observation, 1, True, False, {}

        # Move particles after the turn
        self._move_particles()

        # Switch to the next player
        self.current_player *= -1

        # Update the grid with new particle positions
        self.grid = np.zeros(25, dtype=np.float32)
        for particle in self.particles:
            pos = particle["position"]
            self.grid[pos] += particle["value"]

        observation = self._get_observation()
        return observation, 0, False, False, {}

    def render(self):
        board_str = "Current Grid:\n"
        for row in range(5):
            board_str += "-------------------------\n|"
            for col in range(5):
                idx = row * 5 + col
                cell_value = self.grid[idx]
                if cell_value > 0:
                    board_str += f" {int(cell_value):2d} |"
                else:
                    board_str += "    |"
            board_str += "\n"
        board_str += "-------------------------\n"
        board_str += f"Player 1 Score: {self.scores[1]}\n"
        board_str += f"Player 2 Score: {self.scores[-1]}\n"
        board_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return board_str

    def valid_moves(self):
        return list(range(25))

    def _get_observation(self):
        # Observation consists of the grid, scores, and current player indicator
        obs_grid = self.grid.copy()
        obs_scores = np.array([self.scores[1], self.scores[-1]], dtype=np.float32)
        obs_current_player = np.array([self.current_player], dtype=np.float32)
        observation = np.concatenate([obs_grid, obs_scores, obs_current_player])
        return observation

    def _move_particles(self):
        for particle in self.particles:
            current_pos = particle["position"]
            row, col = divmod(current_pos, 5)
            possible_moves = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 5 and 0 <= new_col < 5:
                        new_pos = new_row * 5 + new_col
                        possible_moves.append(new_pos)
            if possible_moves:
                new_pos = self.np_random.choice(possible_moves)
                particle["position"] = new_pos
