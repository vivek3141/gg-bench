import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 = Attack, 1 = Defend, 2 = Recharge
        self.action_space = spaces.Discrete(3)
        # Observation: [current_player_EP, opponent_EP, current_player_defending, opponent_defending]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([20, 20, 1, 1]), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.EP = [10, 10]  # Energy Points for player 0 and 1
        self.defend_status = [False, False]  # Defend status for player 0 and 1
        self.current_player = 0  # 0 or 1
        self.done = False
        observation = self._get_obs()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}  # Game is over

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self._get_obs(), -10, True, False, {}  # Invalid move

        opponent = 1 - self.current_player

        # Reset defend status for current player at the start of their turn
        self.defend_status[self.current_player] = False

        # Process action
        reward = 0

        if action == 0:  # Attack
            self.EP[self.current_player] -= 2

            if self.defend_status[opponent]:  # Opponent is defending
                damage = 1
            else:
                damage = 3

            self.EP[opponent] -= damage
            if self.EP[opponent] < 0:
                self.EP[opponent] = 0

        elif action == 1:  # Defend
            self.EP[self.current_player] -= 1
            self.defend_status[self.current_player] = True

        elif action == 2:  # Recharge
            self.EP[self.current_player] += 4
            if self.EP[self.current_player] > 20:
                self.EP[self.current_player] = 20

        # Ensure EP doesn't go below 0
        if self.EP[self.current_player] < 0:
            self.EP[self.current_player] = 0

        # Check if opponent's EP is zero (current player wins)
        if self.EP[opponent] == 0:
            self.done = True
            reward = 1
            return self._get_obs(), reward, True, False, {}

        # Switch current player
        self.current_player = opponent

        observation = self._get_obs()

        return (
            observation,
            reward,
            self.done,
            False,
            {},
        )  # observation, reward, terminated, truncated, info

    def render(self):
        s = "Player 0 EP: {}{}".format(
            self.EP[0], " (Defending)" if self.defend_status[0] else ""
        )
        s += "\nPlayer 1 EP: {}{}".format(
            self.EP[1], " (Defending)" if self.defend_status[1] else ""
        )
        s += "\nCurrent Player: Player {}\n".format(self.current_player)
        return s

    def valid_moves(self):
        EP = self.EP[self.current_player]
        valid_actions = []
        if EP >= 2:
            valid_actions.append(0)  # Attack
        if EP >= 1:
            valid_actions.append(1)  # Defend
        valid_actions.append(2)  # Recharge is always valid
        return valid_actions

    def _get_obs(self):
        # Return observation as [current_player_EP, opponent_EP, current_player_defending, opponent_defending]
        opponent = 1 - self.current_player
        obs = [
            self.EP[self.current_player],
            self.EP[opponent],
            1 if self.defend_status[self.current_player] else 0,
            1 if self.defend_status[opponent] else 0,
        ]
        return np.array(obs, dtype=np.float32)
