import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0: Charge, 1: Attack, 2: Special Attack, 3: Shield
        self.action_space = spaces.Discrete(4)

        # Observation space: [current_player_CP, opponent_CP]
        self.observation_space = spaces.Box(low=0, high=5, shape=(2,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_CP = 5
        self.opponent_CP = 5
        self.done = False
        return (
            np.array([self.player_CP, self.opponent_CP], dtype=np.int32),
            {},
        )  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is over, do nothing
            return (
                np.array([self.player_CP, self.opponent_CP], dtype=np.int32),
                0,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Check if action is valid
        if action not in self.valid_moves():
            # Invalid move
            self.done = True
            return (
                np.array([self.player_CP, self.opponent_CP], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Generate opponent's action
        opponent_valid_moves = self.get_opponent_valid_moves()
        opponent_action = np.random.choice(opponent_valid_moves)

        # Apply CP costs
        player_cost = self.get_action_cost(action)
        opponent_cost = self.get_action_cost(opponent_action)

        self.player_CP -= player_cost
        self.opponent_CP -= opponent_cost

        # Apply Charge gains
        if action == 0:  # Charge
            self.player_CP = min(self.player_CP + 1, 5)
        if opponent_action == 0:  # Charge
            self.opponent_CP = min(self.opponent_CP + 1, 5)

        # Apply damage
        player_damage = self.get_action_damage(action)
        opponent_damage = self.get_action_damage(opponent_action)

        # Check for Shield
        player_shielded = action == 3
        opponent_shielded = opponent_action == 3

        # Opponent takes damage from player
        if player_damage > 0 and not opponent_shielded:
            self.opponent_CP -= player_damage

        # Player takes damage from opponent
        if opponent_damage > 0 and not player_shielded:
            self.player_CP -= opponent_damage

        # Ensure CP does not exceed maximum or go below zero
        self.player_CP = max(0, min(5, self.player_CP))
        self.opponent_CP = max(0, min(5, self.opponent_CP))

        # Check win conditions
        if self.opponent_CP <= 0 and self.player_CP <= 0:
            # Both players have CP <= 0, the game continues
            return (
                np.array([self.player_CP, self.opponent_CP], dtype=np.int32),
                0,
                False,
                False,
                {},
            )
        elif self.opponent_CP <= 0:
            # Player wins
            self.done = True
            return (
                np.array([self.player_CP, self.opponent_CP], dtype=np.int32),
                1,
                True,
                False,
                {},
            )
        elif self.player_CP <= 0:
            # Player loses
            self.done = True
            return (
                np.array([self.player_CP, self.opponent_CP], dtype=np.int32),
                -1,
                True,
                False,
                {},
            )
        else:
            # Game continues
            return (
                np.array([self.player_CP, self.opponent_CP], dtype=np.int32),
                0,
                False,
                False,
                {},
            )

    def render(self):
        # Return a string representation of the current game state
        return f"Player CP: {self.player_CP}, Opponent CP: {self.opponent_CP}"

    def valid_moves(self):
        valid_actions = [0]  # Charge is always possible
        cp = self.player_CP
        if cp >= 1:
            valid_actions.append(1)  # Attack
            valid_actions.append(3)  # Shield
        if cp >= 3:
            valid_actions.append(2)  # Special Attack
        return valid_actions

    def get_opponent_valid_moves(self):
        valid_actions = [0]  # Charge is always possible
        cp = self.opponent_CP
        if cp >= 1:
            valid_actions.append(1)  # Attack
            valid_actions.append(3)  # Shield
        if cp >= 3:
            valid_actions.append(2)  # Special Attack
        return valid_actions

    def get_action_cost(self, action):
        if action == 0:  # Charge
            return 0
        elif action == 1:  # Attack
            return 1
        elif action == 2:  # Special Attack
            return 3
        elif action == 3:  # Shield
            return 1
        else:
            return 0  # Invalid action (should not happen)

    def get_action_damage(self, action):
        if action == 1:  # Attack
            return 1
        elif action == 2:  # Special Attack
            return 3
        else:
            return 0  # Charge or Shield deals no damage
