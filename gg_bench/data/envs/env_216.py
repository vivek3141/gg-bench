import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space

        # The action is an integer from 0 to 99, representing bids from 1 to 100
        self.action_space = spaces.Discrete(100)  # Actions from 0 to 99

        # Observation space: [tokens_player, tokens_opponent, victory_points_player, victory_points_opponent]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([100, 100, 5, 5]), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tokens_player = 100
        self.tokens_opponent = 100
        self.victory_points_player = 0
        self.victory_points_opponent = 0
        self.done = False
        # Return observation and info
        observation = np.array(
            [
                self.tokens_player,
                self.tokens_opponent,
                self.victory_points_player,
                self.victory_points_opponent,
            ],
            dtype=np.int32,
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        # Convert action to bid
        bid_player = action + 1  # Since action ranges from 0 to 99, bids from 1 to 100

        # Check if bid is valid
        if bid_player < 1 or bid_player > self.tokens_player:
            self.done = True
            return self.get_observation(), -10, True, False, {}

        # Generate opponent's bid (random valid bid)
        bid_opponent = np.random.randint(1, self.tokens_opponent + 1)

        # Both players subtract their bid amounts from their total tokens
        self.tokens_player -= bid_player
        self.tokens_opponent -= bid_opponent

        # Determine winner of the round
        if bid_player > bid_opponent:
            self.victory_points_player += 1
        elif bid_player < bid_opponent:
            self.victory_points_opponent += 1
        else:
            # Tie, no victory points awarded
            pass

        # Check for victory conditions
        if self.victory_points_player >= 5 and self.victory_points_opponent >= 5:
            if self.tokens_player > self.tokens_opponent:
                self.done = True
                return self.get_observation(), 1, True, False, {}
            elif self.tokens_player < self.tokens_opponent:
                self.done = True
                return self.get_observation(), -1, True, False, {}
            else:
                if self.tokens_player <= 0 and self.tokens_opponent <= 0:
                    # Both players run out of tokens, game tied
                    self.done = True
                    return self.get_observation(), 0, True, False, {}
                else:
                    # Game continues
                    return self.get_observation(), 0, False, False, {}
        elif self.victory_points_player >= 5:
            self.done = True
            return self.get_observation(), 1, True, False, {}
        elif self.victory_points_opponent >= 5:
            self.done = True
            return self.get_observation(), -1, True, False, {}
        elif self.tokens_player <= 0 and self.tokens_opponent > 0:
            # Player ran out of tokens, cannot bid next round, loses
            self.done = True
            return self.get_observation(), -1, True, False, {}
        elif self.tokens_opponent <= 0 and self.tokens_player > 0:
            # Opponent ran out of tokens, player wins
            self.done = True
            return self.get_observation(), 1, True, False, {}
        elif self.tokens_player <= 0 and self.tokens_opponent <= 0:
            # Both players ran out of tokens, compare victory points
            if self.victory_points_player > self.victory_points_opponent:
                self.done = True
                return self.get_observation(), 1, True, False, {}
            elif self.victory_points_player < self.victory_points_opponent:
                self.done = True
                return self.get_observation(), -1, True, False, {}
            else:
                # Tie
                self.done = True
                return self.get_observation(), 0, True, False, {}
        else:
            # Game continues
            return self.get_observation(), 0, False, False, {}

    def get_observation(self):
        return np.array(
            [
                self.tokens_player,
                self.tokens_opponent,
                self.victory_points_player,
                self.victory_points_opponent,
            ],
            dtype=np.int32,
        )

    def render(self):
        state_str = f"Player Tokens: {self.tokens_player} | Victory Points: {self.victory_points_player}\n"
        state_str += f"Opponent Tokens: {self.tokens_opponent} | Victory Points: {self.victory_points_opponent}\n"
        return state_str

    def valid_moves(self):
        # Returns list of valid actions (i.e., bids the player can make)
        return [i for i in range(self.tokens_player)]  # Since action = bid -1
