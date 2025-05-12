import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space
        # Actions:
        # 0-4: No defense token, sea level rise 1-5
        # 5-9: Use defense token, sea level rise 1-5
        self.action_space = spaces.Discrete(10)

        # Define observation space
        # Observation: [Player1_island_units, Player2_island_units, sea_level, Player1_defense_tokens, Player2_defense_tokens]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 1, 0, 0], dtype=np.int32),
            high=np.array([30, 30, 100, 2, 2], dtype=np.int32),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1_island_units = 30
        self.player2_island_units = 30
        self.sea_level = 1
        self.player1_defense_tokens = 2
        self.player2_defense_tokens = 2
        self.current_player = 1
        self.done = False

        obs = self.get_observation()
        return obs, {}

    def get_observation(self):
        obs = np.array(
            [
                self.player1_island_units,
                self.player2_island_units,
                self.sea_level,
                self.player1_defense_tokens,
                self.player2_defense_tokens,
            ],
            dtype=np.int32,
        )
        return obs

    def step(self, action):
        if self.done:
            return self.get_observation(), -10, True, False, {}

        # Check if action is valid
        if action < 0 or action >= 10:
            # Invalid action
            self.done = True
            return self.get_observation(), -10, True, False, {}

        # Map action index to defense token usage and sea level rise amount
        if action <= 4:
            use_defense_token = False
            sea_level_rise = action + 1  # action 0 corresponds to sea level rise 1
        else:
            use_defense_token = True
            sea_level_rise = action - 5 + 1  # action 5 corresponds to sea level rise 1

        # Check if player can use a defense token if they chose to
        if use_defense_token:
            if self.current_player == 1:
                if self.player1_defense_tokens <= 0:
                    # Invalid action: no defense tokens remaining
                    self.done = True
                    return self.get_observation(), -10, True, False, {}
                else:
                    self.player1_defense_tokens -= 1
            else:  # current_player == 2
                if self.player2_defense_tokens <= 0:
                    # Invalid action: no defense tokens remaining
                    self.done = True
                    return self.get_observation(), -10, True, False, {}
                else:
                    self.player2_defense_tokens -= 1

        # Increase sea level
        self.sea_level += sea_level_rise

        # Apply sea level impact to both islands
        sea_level_impact_p1 = self.sea_level
        sea_level_impact_p2 = self.sea_level

        if self.current_player == 1 and use_defense_token:
            sea_level_impact_p1 = max(0, self.sea_level - 2)
        if self.current_player == 2 and use_defense_token:
            sea_level_impact_p2 = max(0, self.sea_level - 2)

        # Reduce island units
        self.player1_island_units -= sea_level_impact_p1
        self.player2_island_units -= sea_level_impact_p2

        # Ensure island units do not go below zero
        self.player1_island_units = max(0, self.player1_island_units)
        self.player2_island_units = max(0, self.player2_island_units)

        # Check for end-of-game conditions
        if self.player1_island_units <= 0 or self.player2_island_units <= 0:
            self.done = True
            # Determine winner
            if self.player1_island_units <= 0 and self.player2_island_units <= 0:
                # Both islands sank
                # According to the rules, the current player loses
                if self.current_player == 1:
                    reward = -1  # Current player loses
                else:
                    reward = 1  # Current player wins
            elif self.player1_island_units <= 0:
                # Player 1's island sank
                if self.current_player == 1:
                    reward = -1  # Current player loses
                else:
                    reward = 1  # Current player wins
            elif self.player2_island_units <= 0:
                # Player 2's island sank
                if self.current_player == 2:
                    reward = -1  # Current player loses
                else:
                    reward = 1  # Current player wins
        else:
            # Game continues
            reward = 0
            # Switch current player
            self.current_player = 2 if self.current_player == 1 else 1

        return self.get_observation(), reward, self.done, False, {}

    def render(self):
        state_str = "---- Game State ----\n"
        state_str += f"Player 1 Island Units: {self.player1_island_units}\n"
        state_str += f"Player 2 Island Units: {self.player2_island_units}\n"
        state_str += f"Sea Level: {self.sea_level}\n"
        state_str += f"Player 1 Defense Tokens: {self.player1_defense_tokens}\n"
        state_str += f"Player 2 Defense Tokens: {self.player2_defense_tokens}\n"
        state_str += f"Current Player: Player {self.current_player}\n"
        state_str += "---------------------\n"
        return state_str

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        for action in range(10):
            if action <= 4:
                # No defense token required
                valid_actions.append(action)
            else:
                # Defense token required
                if self.current_player == 1 and self.player1_defense_tokens > 0:
                    valid_actions.append(action)
                elif self.current_player == 2 and self.player2_defense_tokens > 0:
                    valid_actions.append(action)
        return valid_actions
