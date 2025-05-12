import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 25 cells in the 5x5 grid
        self.action_space = spaces.Discrete(25)

        # Observation space consists of the grid cells, player scores, and current player indicator
        # - Grid cells: 25 values
        # - Player scores: 2 values
        # - Current player: 1 value
        self.observation_space = spaces.Box(low=0, high=15, shape=(28,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the grid with values from 1 to 5
        self.cell_values = np.array([1, 2, 3, 4, 5] * 5)
        # Shuffle the cell values to randomize the grid
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.cell_values = self.np_random.permutation(self.cell_values)
        # Initialize cell statuses: 0 (unclaimed), 1 (Player 1 claimed), 2 (Player 2 claimed)
        self.cell_status = np.zeros(25, dtype=np.int32)
        # Player scores: index 0 for Player 1, index 1 for Player 2
        self.player_scores = [0, 0]
        # Current player: 0 for Player 1, 1 for Player 2
        self.current_player = 0
        # Game over flag
        self.done = False
        # Return the initial observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if the action is valid
        if action not in self.valid_moves():
            # Invalid move results in immediate loss
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Claim the selected cell
        self.cell_status[action] = self.current_player + 1  # 1 or 2
        # Update the current player's score
        self.player_scores[self.current_player] += self.cell_values[action]

        # Check for win condition
        if self.player_scores[self.current_player] == 15:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, {}

        # Check for loss condition
        elif self.player_scores[self.current_player] > 15:
            # Current player loses
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        else:
            # Switch to the next player
            self.current_player = 1 - self.current_player
            reward = 0
            return self._get_observation(), reward, False, False, {}

    def render(self):
        # Generate a visual representation of the grid and scores
        grid = ""
        for i in range(5):
            row = ""
            for j in range(5):
                idx = i * 5 + j
                status = self.cell_status[idx]
                if status == 0:
                    # Unclaimed cell, display the value
                    row += f"[{self.cell_values[idx]}]"
                elif status == 1:
                    # Claimed by Player 1
                    row += "[X]"
                elif status == 2:
                    # Claimed by Player 2
                    row += "[O]"
            grid += row + "\n"
        # Display the scores and current player
        output = f"Player 1 Score: {self.player_scores[0]}\n"
        output += f"Player 2 Score: {self.player_scores[1]}\n\n"
        output += "Grid:\n"
        output += grid
        output += f"Player {self.current_player + 1}'s Turn."
        print(output)

    def valid_moves(self):
        # Generate a list of valid moves
        valid_actions = []
        current_score = self.player_scores[self.current_player]
        for idx in range(25):
            if self.cell_status[idx] == 0:
                potential_score = current_score + self.cell_values[idx]
                if potential_score <= 15:
                    valid_actions.append(idx)
        return valid_actions

    def _get_observation(self):
        # Construct the observation array
        observation = np.zeros(28, dtype=np.int32)
        # Cells status and values
        for idx in range(25):
            status = self.cell_status[idx]
            if status == 0:
                # Unclaimed cell, store its value (1 to 5)
                observation[idx] = self.cell_values[idx]
            elif status == 1:
                # Claimed by Player 1
                observation[idx] = 6
            elif status == 2:
                # Claimed by Player 2
                observation[idx] = 7
        # Player scores
        observation[25] = self.player_scores[0]
        observation[26] = self.player_scores[1]
        # Current player indicator
        observation[27] = self.current_player
        return observation
