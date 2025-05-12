import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Move Forward, 1-10 - Use Detector on Step N
        self.action_space = spaces.Discrete(11)  # Actions 0 to 10

        # Observation space:
        # Steps 1-10: -1 (mine diffused), 0 (unknown), 1 (confirmed safe)
        # Player positions: own position, opponent position (0 to 10)
        # Detectors remaining: own detectors, opponent detectors (0 to 2)
        # Total size: 14
        self.observation_space = spaces.Box(
            low=np.array([-1] * 10 + [0, 0, 0, 0]),
            high=np.array([1] * 10 + [10, 10, 2, 2]),
            dtype=np.int8,
        )

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly place mines on three of the ten steps (steps 1 to 10)
        self.mines = self.np_random.choice(range(1, 11), size=3, replace=False)
        self.mines.sort()
        self.mines = list(self.mines)

        # Initialize steps: -1 (mine diffused), 0 (unknown), 1 (confirmed safe)
        self.steps = np.zeros(10, dtype=np.int8)

        # Initialize player positions (index 0 and 1)
        self.player_positions = [0, 0]

        # Initialize detectors remaining for each player
        self.detectors_remaining = [2, 2]

        # Game state
        self.current_player = 0  # 0 for Player 1, 1 for Player 2

        self.done = False

        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        # Combine self.steps, positions, detectors remaining
        obs = np.concatenate(
            (
                self.steps,
                np.array(
                    [
                        self.player_positions[0],
                        self.player_positions[1],
                        self.detectors_remaining[0],
                        self.detectors_remaining[1],
                    ],
                    dtype=np.int8,
                ),
            )
        )
        return obs

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}
        reward = 0
        info = {}

        # Get current player
        player = self.current_player
        opponent = 1 - player

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, info

        # Perform action
        if action == 0:
            # Move forward one step
            new_position = self.player_positions[player] + 1
            self.player_positions[player] = new_position

            # Check for mine
            if new_position in self.mines and self.steps[new_position - 1] != -1:
                # Stepped on a mine and it was not diffused
                self.done = True
                reward = -1  # Player loses
                return self._get_observation(), reward, True, False, info

            # Check if reached step 10
            if new_position == 10:
                # Player wins
                self.done = True
                reward = 1
                return self._get_observation(), reward, True, False, info

        else:
            # Use detector on step N
            N = action  # N from 1 to 10

            # Use a detector
            self.detectors_remaining[player] -= 1

            # Check for mine
            if N in self.mines:
                # Diffuse mine
                self.steps[N - 1] = -1
            else:
                # Confirmed safe
                self.steps[N - 1] = 1

        # Switch current player
        self.current_player = opponent

        return self._get_observation(), reward, False, False, info

    def render(self):
        # Return a visual representation of the environment state as a string
        s = "Minefield Navigator Game State:\n"
        s += "Steps:\n"
        for i in range(10):
            s += f" Step {i+1}: "
            if self.steps[i] == -1:
                s += "Mine diffused\n"
            elif self.steps[i] == 1:
                s += "Confirmed safe\n"
            else:
                s += "Unknown\n"
        s += "\n"
        s += f"Player 1 Position: {self.player_positions[0]}, Detectors Remaining: {self.detectors_remaining[0]}\n"
        s += f"Player 2 Position: {self.player_positions[1]}, Detectors Remaining: {self.detectors_remaining[1]}\n"
        s += f"Current Turn: Player {self.current_player + 1}\n"
        return s

    def valid_moves(self):
        # Return list of valid actions for the current player
        player = self.current_player
        actions = []

        # Move forward is always valid
        actions.append(0)

        # Use detector actions
        if self.detectors_remaining[player] > 0:
            current_position = self.player_positions[player]
            # Detector can only scan steps ahead of the player's current position
            for N in range(current_position + 1, 11):
                actions.append(N)

        return actions
