import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Attack, 1 - Defend, 2 - Recharge
        self.action_space = spaces.Discrete(3)

        # Observation space: [Current HP, EP, Defend, Opponent HP, EP, Defend]
        low = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
        high = np.array([10, 5, 1, 10, 5, 1], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize HP, EP, Defend status for both players
        self.HP = np.array([10, 10], dtype=np.int32)
        self.EP = np.array([5, 5], dtype=np.int32)
        self.Defend = np.array([0, 0], dtype=np.int32)
        self.current_player = 0
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.array(
            [
                self.HP[self.current_player],
                self.EP[self.current_player],
                self.Defend[self.current_player],
                self.HP[1 - self.current_player],
                self.EP[1 - self.current_player],
                self.Defend[1 - self.current_player],
            ],
            dtype=np.int32,
        )
        return obs

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # At the start of the player's turn, their Defend status expires
        self.Defend[self.current_player] = 0

        # Check if action is valid
        if action not in self.valid_moves():
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        reward = 0

        # Process action
        if action == 0:  # Attack
            # Spend EP
            self.EP[self.current_player] -= 1
            # Check if opponent is defending
            if self.Defend[1 - self.current_player]:
                # Opponent's Defend negates the attack
                self.Defend[1 - self.current_player] = 0
            else:
                # Opponent takes 2 damage
                self.HP[1 - self.current_player] -= 2
                # Check if opponent's HP <= 0
                if self.HP[1 - self.current_player] <= 0:
                    reward = 1
                    self.done = True

        elif action == 1:  # Defend
            # Spend EP
            self.EP[self.current_player] -= 1
            # Set Defend status
            self.Defend[self.current_player] = 1

        elif action == 2:  # Recharge
            # Restore 2 EP, not to exceed 5
            self.EP[self.current_player] = min(5, self.EP[self.current_player] + 2)

        # If the game is not over, switch players
        if not self.done:
            # Switch current player
            self.current_player = 1 - self.current_player
            # Defend status of the new current player will expire at the start of their turn
        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        state_str = f"Player {self.current_player + 1}'s turn.\n"
        state_str += f"Player 1 - HP: {self.HP[0]}, EP: {self.EP[0]}"
        if self.Defend[0]:
            state_str += " (Defending)\n"
        else:
            state_str += "\n"
        state_str += f"Player 2 - HP: {self.HP[1]}, EP: {self.EP[1]}"
        if self.Defend[1]:
            state_str += " (Defending)\n"
        else:
            state_str += "\n"
        return state_str

    def valid_moves(self):
        valid_actions = []
        curr_ep = self.EP[self.current_player]
        # Attack
        if curr_ep >= 1:
            valid_actions.append(0)
        # Defend
        if curr_ep >= 1:
            valid_actions.append(1)
        # Recharge
        valid_actions.append(2)
        return valid_actions
