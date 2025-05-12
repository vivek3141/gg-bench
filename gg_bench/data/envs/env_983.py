import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 25 possible actions (5 attacking soldiers x 5 defending soldiers)
        self.action_space = spaces.Discrete(25)

        # Observation space is an array of 10 elements:
        # [player's soldiers (1-5), opponent's soldiers (1-5)]
        # Each element is 1 if the soldier is available or 0 if eliminated
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Both players start with soldiers 1-5 available (represented by 1)
        self.player_soldiers = [1, 1, 1, 1, 1]  # Current player's soldiers
        self.opponent_soldiers = [1, 1, 1, 1, 1]  # Opponent's soldiers

        # Game not done
        self.done = False

        # Return initial observation and info
        observation = np.array(
            self.player_soldiers + self.opponent_soldiers, dtype=np.int8
        )
        return observation, {}

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Decode action into attacking soldier and defending soldier (both 1-5)
        attacking_soldier_number = action // 5 + 1
        defending_soldier_number = action % 5 + 1

        attacking_soldier_index = attacking_soldier_number - 1
        defending_soldier_index = defending_soldier_number - 1

        # Check if both soldiers are available
        if (
            self.player_soldiers[attacking_soldier_index] == 0
            or self.opponent_soldiers[defending_soldier_index] == 0
        ):
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Resolve attack
        if attacking_soldier_number > defending_soldier_number:
            # Attacker wins; defender's soldier is eliminated
            self.opponent_soldiers[defending_soldier_index] = 0
        elif attacking_soldier_number == defending_soldier_number:
            # Tie; both soldiers are eliminated
            self.player_soldiers[attacking_soldier_index] = 0
            self.opponent_soldiers[defending_soldier_index] = 0
        else:
            # Defender wins; attacker's soldier is eliminated
            self.player_soldiers[attacking_soldier_index] = 0

        # Attacking soldier is always removed after use
        self.player_soldiers[attacking_soldier_index] = 0

        # Check for game over conditions
        player_soldiers_left = sum(self.player_soldiers)
        opponent_soldiers_left = sum(self.opponent_soldiers)

        if player_soldiers_left == 0 and opponent_soldiers_left == 0:
            # Both players have no soldiers left; attacking player loses
            self.done = True
            return self._get_obs(), -1, True, False, {}
        elif opponent_soldiers_left == 0:
            # Opponent has no soldiers left; current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}
        elif player_soldiers_left == 0:
            # Current player has no soldiers left; current player loses
            self.done = True
            return self._get_obs(), -1, True, False, {}

        # Switch turns: Swap player and opponent soldiers
        self.player_soldiers, self.opponent_soldiers = (
            self.opponent_soldiers,
            self.player_soldiers,
        )

        # Return observation, reward, done, truncated, and info
        return self._get_obs(), 0, False, False, {}

    def render(self):
        # Return a string representation of the game state
        player_soldiers_list = [
            i + 1 for i, s in enumerate(self.player_soldiers) if s == 1
        ]
        opponent_soldiers_list = [
            i + 1 for i, s in enumerate(self.opponent_soldiers) if s == 1
        ]
        s = "Player's soldiers: {}\n".format(player_soldiers_list)
        s += "Opponent's soldiers: {}\n".format(opponent_soldiers_list)
        return s

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        for attacking_soldier_number in range(1, 6):
            for defending_soldier_number in range(1, 6):
                attacking_soldier_index = attacking_soldier_number - 1
                defending_soldier_index = defending_soldier_number - 1
                if (
                    self.player_soldiers[attacking_soldier_index] == 1
                    and self.opponent_soldiers[defending_soldier_index] == 1
                ):
                    action = (attacking_soldier_number - 1) * 5 + (
                        defending_soldier_number - 1
                    )
                    valid_actions.append(action)
        return valid_actions

    def _get_obs(self):
        # Return the current observation
        return np.array(self.player_soldiers + self.opponent_soldiers, dtype=np.int8)
