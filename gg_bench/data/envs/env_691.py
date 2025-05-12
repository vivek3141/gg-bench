import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions:
        # 0 - Charge
        # 1 - Attack with 1 EP
        # 2 - Attack with 2 EP
        # 3 - Attack with 3 EP
        self.action_space = spaces.Discrete(4)

        # Observation space: [Player 1 EP, Player 2 EP]
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1_ep = 10
        self.player2_ep = 10
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        observation = np.array([self.player1_ep, self.player2_ep], dtype=np.int32)
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}  # Game is over

        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Get current player's EP
        if self.current_player == 1:
            current_ep = self.player1_ep
            opponent_ep = self.player2_ep
        else:
            current_ep = self.player2_ep
            opponent_ep = self.player1_ep

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            reward = -10
            terminated = True
            self.done = True
            return self._get_obs(), reward, terminated, truncated, info

        # Execute action
        if action == 0:
            # Charge action
            current_ep += 2
            if current_ep > 10:
                current_ep = 10
        else:
            # Attack action
            attack_cost = action  # action corresponds to attack cost (1, 2, or 3)
            current_ep -= attack_cost
            damage = attack_cost * 2
            opponent_ep -= damage

            # Check for win conditions
            if opponent_ep <= 0:
                # Attacking player wins
                reward = 1
                terminated = True
                self.done = True
                if self.current_player == 1:
                    self.player1_ep = current_ep
                    self.player2_ep = max(opponent_ep, 0)
                else:
                    self.player2_ep = current_ep
                    self.player1_ep = max(opponent_ep, 0)
                return self._get_obs(), reward, terminated, truncated, info

            if current_ep <= 0:
                if opponent_ep <= 0:
                    # Both EPs <= 0, attacking player wins
                    reward = 1
                    terminated = True
                    self.done = True
                    if self.current_player == 1:
                        self.player1_ep = max(current_ep, 0)
                        self.player2_ep = max(opponent_ep, 0)
                    else:
                        self.player2_ep = max(current_ep, 0)
                        self.player1_ep = max(opponent_ep, 0)
                    return self._get_obs(), reward, terminated, truncated, info

        # Update EPs
        if self.current_player == 1:
            self.player1_ep = current_ep
            self.player2_ep = opponent_ep
        else:
            self.player2_ep = current_ep
            self.player1_ep = opponent_ep

        # Check if current player reduced their own EP to zero without winning
        if current_ep <= 0:
            reward = -10
            terminated = True
            self.done = True
            return self._get_obs(), reward, terminated, truncated, info

        # Switch current player
        self.current_player *= -1

        observation = self._get_obs()
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return np.array([self.player1_ep, self.player2_ep], dtype=np.int32)

    def render(self):
        return f"Player 1 EP: {self.player1_ep}, Player 2 EP: {self.player2_ep}, Current Player: {'1' if self.current_player == 1 else '2'}"

    def valid_moves(self):
        # Returns list of valid actions for the current player
        if self.done:
            return []

        if self.current_player == 1:
            current_ep = self.player1_ep
        else:
            current_ep = self.player2_ep

        valid_actions = [0]  # Charge is always valid

        if current_ep >= 1:
            valid_actions.append(1)
        if current_ep >= 2:
            valid_actions.append(2)
        if current_ep >= 3:
            valid_actions.append(3)

        return valid_actions
