import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 5 elements/actions: Fire, Water, Earth, Air, Lightning
        self.action_space = spaces.Discrete(5)

        # Observation space: Player's HP and Opponent's HP
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.int32)

        # Elements mapping
        self.elements = ["Fire", "Water", "Earth", "Air", "Lightning"]

        # Strengths: Each element beats certain other elements
        self.strengths = {
            0: [2, 3],  # Fire beats Earth and Air
            1: [0, 4],  # Water beats Fire and Lightning
            2: [1, 4],  # Earth beats Water and Lightning
            3: [2, 1],  # Air beats Earth and Water
            4: [0, 3],  # Lightning beats Fire and Air
        }

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_hp = [10, 10]  # [Current Player HP, Opponent HP]
        self.done = False
        return np.array(self.player_hp, dtype=np.int32), {}  # Observation, info

    def step(self, action):
        # Validate the action
        if not self.action_space.contains(action):
            self.done = True
            return (
                np.array(self.player_hp, dtype=np.int32),
                -10,  # Invalid move penalty
                True,  # Terminated
                False,  # Truncated
                {},  # Info
            )

        if self.done:
            return (
                np.array(self.player_hp, dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Opponent selects an element randomly
        opponent_action = self.action_space.sample()

        # Resolve the clash
        player_element = action
        opponent_element = opponent_action

        player_damage = 0
        opponent_damage = 0

        if player_element == opponent_element:
            # Draw: Both players take 1 damage
            player_damage = 1
            opponent_damage = 1
        elif opponent_element in self.strengths[player_element]:
            # Current player wins the clash
            opponent_damage = 3
        elif player_element in self.strengths[opponent_element]:
            # Opponent wins the clash
            player_damage = 3
        else:
            # No damage
            pass

        # Update health points
        self.player_hp[0] -= player_damage
        self.player_hp[1] -= opponent_damage

        # Check for game over conditions
        if self.player_hp[1] <= 0:
            self.done = True
            reward = 1  # Current player wins
        elif self.player_hp[0] <= 0:
            self.done = True
            reward = -1  # Current player loses
        else:
            reward = 0  # Continue playing

        observation = np.array(self.player_hp, dtype=np.int32)
        info = {}

        return observation, reward, self.done, False, info

    def render(self):
        return f"Player HP: {self.player_hp[0]}, Opponent HP: {self.player_hp[1]}"

    def valid_moves(self):
        return [i for i in range(5)]  # All moves (0 to 4) are valid
