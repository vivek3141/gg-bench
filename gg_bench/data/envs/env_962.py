import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            5
        )  # Positions 0-4 (representing positions 1-5)
        self.observation_space = spaces.Box(low=-1, high=5, shape=(12,), dtype=np.int32)

        self._rng = np.random.default_rng()

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        # Initialize positions for both players
        self.positions = [
            self.allocate_defense_points(),
            self.allocate_defense_points(),
        ]
        # Known opponent positions for each player
        self.known_positions = [[-1] * 5, [-1] * 5]
        # Last positions attacked by each player
        self.last_attacks = [-1, -1]  # -1 indicates no attack yet
        # Current player (0 or 1)
        self.current_player = 0
        # Game over flag
        self.done = False
        return self.get_observation(), {}  # Return observation and info

    def allocate_defense_points(self):
        points = 5
        positions = [0] * 5
        for i in range(4):
            max_allocatable = min(5, points)
            dp = self._rng.integers(
                0, max_allocatable + 1
            )  # Random DP between 0 and points left
            positions[i] = dp
            points -= dp
        positions[4] = points  # Assign remaining points to the last position
        self._rng.shuffle(positions)
        return positions

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}
        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self.get_observation(), -10, True, False, {}
        # Perform action
        opponent = 1 - self.current_player
        # Reduce DP of opponent's position
        self.positions[opponent][action] -= 1
        if self.positions[opponent][action] < 0:
            self.positions[opponent][action] = 0
        # Update known positions
        self.known_positions[self.current_player][action] = self.positions[opponent][
            action
        ]
        # Update last_attack
        self.last_attacks[self.current_player] = action

        # Check for victory
        if all(dp == 0 for dp in self.positions[opponent]):
            self.done = True
            return self.get_observation(), 1, True, False, {}

        # Switch to next player
        self.current_player = opponent
        return self.get_observation(), 0, False, False, {}

    def get_observation(self):
        # Current player's own positions
        own_positions = self.positions[self.current_player]
        # Opponent's known positions
        opponent_positions = self.known_positions[self.current_player]
        # Last attack made by current player
        last_attack = self.last_attacks[self.current_player]
        # Last attack made by opponent
        opponent_last_attack = self.last_attacks[1 - self.current_player]
        observation = np.array(
            own_positions + opponent_positions + [last_attack, opponent_last_attack],
            dtype=np.int32,
        )
        return observation

    def valid_moves(self):
        opponent = 1 - self.current_player
        valid_actions = []
        for i in range(5):
            if self.positions[opponent][i] > 0:
                valid_actions.append(i)
        # Remove last attacked position if needed
        last_attack = self.last_attacks[self.current_player]
        if len(valid_actions) > 1 and last_attack in valid_actions:
            valid_actions.remove(last_attack)
        return valid_actions

    def render(self):
        opponent = 1 - self.current_player
        # Display current player's view
        own_positions = self.positions[self.current_player]
        opponent_known = self.known_positions[self.current_player]
        render_str = f"Current player: {self.current_player + 1}\n"
        render_str += "Your positions:\n"
        for i, dp in enumerate(own_positions):
            render_str += f"  Position {i+1}: {dp} DP\n"
        render_str += "Opponent's known positions:\n"
        for i, dp in enumerate(opponent_known):
            if dp == -1:
                render_str += f"  Position {i+1}: Unknown\n"
            else:
                render_str += f"  Position {i+1}: {dp} DP\n"
        if self.last_attacks[self.current_player] != -1:
            render_str += f"Last position you attacked: Position {self.last_attacks[self.current_player]+1}\n"
        else:
            render_str += "Last position you attacked: None\n"
        if self.last_attacks[opponent] != -1:
            render_str += f"Last position opponent attacked: Position {self.last_attacks[opponent]+1}\n"
        else:
            render_str += "Last position opponent attacked: None\n"
        return render_str
