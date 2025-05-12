import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0: Attack, 1: Bypass

        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(2,),
            dtype=np.int32,  # [Current player's firewall, Opponent's firewall]
        )

        # Initialize the state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize firewall strengths
        self.firewall_strength = {1: 10, -1: 10}

        # Set current player (1 or -1)
        self.current_player = 1

        # Game state
        self.done = False

        # Initialize random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        return self._get_observation(), {}  # observation, info

    def step(self, action):
        if action not in self.valid_moves():
            # Invalid action
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, self.done, False, {}

        if self.done:
            # Game is already over
            observation = self._get_observation()
            return observation, 0, self.done, False, {}

        opponent = -self.current_player

        # Process action
        if action == 0:  # Attack
            roll = self.np_random.integers(1, 7)  # Roll between 1 and 6 inclusive
            if roll >= 4:
                # Attack successful
                self.firewall_strength[opponent] -= 3
                self.firewall_strength[opponent] = max(
                    0, self.firewall_strength[opponent]
                )
        elif action == 1:  # Bypass
            # Guaranteed damage
            self.firewall_strength[opponent] -= 1
            self.firewall_strength[opponent] = max(0, self.firewall_strength[opponent])

        # Check for win condition
        if self.firewall_strength[opponent] == 0:
            # Current player wins
            reward = 1
            self.done = True
        else:
            # No win yet, switch turns
            self.current_player = opponent
            reward = 0

        observation = self._get_observation()
        return observation, reward, self.done, False, {}

    def render(self):
        output = f"Player {1 if self.current_player == 1 else 2}'s Turn:\n"
        output += f"Your Firewall: {self.firewall_strength[self.current_player]}\n"
        output += (
            f"Opponent's Firewall: {self.firewall_strength[-self.current_player]}\n"
        )
        return output

    def valid_moves(self):
        # Both actions are always valid
        return [0, 1]

    def _get_observation(self):
        return np.array(
            [
                self.firewall_strength[self.current_player],
                self.firewall_strength[-self.current_player],
            ],
            dtype=np.int32,
        )
